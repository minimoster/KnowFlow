// rag-export reads existing evaluation backends. It never bootstraps schemas,
// indexes, collections, ingestion, the application server or a sandbox.
package main

import (
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"runtime/debug"
	"strings"
	"time"

	"agi-assistant/config"
	"agi-assistant/internal/domain/rag"
	"agi-assistant/internal/infrastructure/llm"
	"agi-assistant/internal/infrastructure/persistence/ragchunk"
	"agi-assistant/internal/pkg/logger"
	es "github.com/elastic/go-elasticsearch/v8"
	"github.com/lib/pq"
	milvus "github.com/milvus-io/milvus-sdk-go/v2/client"
)

type backendConfig struct {
	PGDSNEnv         string   `json:"pg_dsn_env"`
	MilvusAddress    string   `json:"milvus_address"`
	ESAddresses      []string `json:"es_addresses"`
	ESUsernameEnv    string   `json:"es_username_env"`
	ESPasswordEnv    string   `json:"es_password_env"`
	EmbeddingURL     string   `json:"embedding_url"`
	EmbeddingKeyEnv  string   `json:"embedding_key_env"`
	EmbeddingModel   string   `json:"embedding_model"`
	RerankURL        string   `json:"rerank_url"`
	RerankKeyEnv     string   `json:"rerank_key_env"`
	RerankModel      string   `json:"rerank_model"`
	RerankPreviewLen int      `json:"rerank_preview_len"`
}

type request struct {
	Question   string          `json:"question"`
	Mode       string          `json:"mode"`
	CandidateK int             `json:"candidate_k"`
	RRFK       int             `json:"rrf_k"`
	Expected   *expectedConfig `json:"expected_config,omitempty"`
}

type expectedConfig struct {
	Revision         string  `json:"revision"`
	EmbeddingModel   string  `json:"embedding_model"`
	Reranker         *string `json:"reranker"`
	RerankPreviewLen int     `json:"rerank_preview_len"`
}

func checkExpected(r request, backend backendConfig, revision string) error {
	e := r.Expected
	if e == nil || e.Revision != revision || e.EmbeddingModel != backend.EmbeddingModel || e.RerankPreviewLen != backend.RerankPreviewLen {
		return errors.New("expected build/model configuration mismatch")
	}
	if r.Mode == "hybrid_rerank" {
		if e.Reranker == nil || *e.Reranker != backend.RerankModel {
			return errors.New("expected reranker mismatch")
		}
	} else if e.Reranker != nil {
		return errors.New("unexpected reranker")
	}
	return nil
}

type hit struct {
	PGID          int64  `json:"pg_id"`
	ContentSHA256 string `json:"content_sha256"`
	DocHash       string `json:"doc_hash"`
	ChunkIdx      int    `json:"chunk_idx"`
}

type retriever interface {
	RetrieveForEvaluation(context.Context, string, string, int) ([]rag.HybridResult, error)
}

func validateRequest(r request) error {
	if strings.TrimSpace(r.Question) == "" || r.CandidateK < 5 || r.CandidateK > 1000 || r.RRFK < 1 || r.RRFK > 100000 {
		return errors.New("invalid request")
	}
	if r.Mode != "vector" && r.Mode != "hybrid" && r.Mode != "hybrid_rerank" {
		return errors.New("invalid mode")
	}
	return nil
}

// Child text is fingerprinted before any parent-content presentation can replace it.
func retrieve(ctx context.Context, r request, engine retriever) ([]hit, error) {
	if err := validateRequest(r); err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	rows, err := engine.RetrieveForEvaluation(ctx, r.Question, r.Mode, r.CandidateK)
	if err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(rows) > 5 {
		return nil, errors.New("too many hits")
	}
	seen := map[int64]bool{}
	result := make([]hit, 0, len(rows))
	for _, row := range rows {
		if row.PGID <= 0 || seen[row.PGID] || row.Chunk.Content == "" {
			return nil, errors.New("invalid child hit")
		}
		seen[row.PGID] = true
		result = append(result, hit{PGID: row.PGID, ContentSHA256: fmt.Sprintf("%x", sha256.Sum256([]byte(row.Chunk.Content)))})
	}
	return result, nil
}

type snapshotRow struct {
	PGID     int64  `json:"pg_id"`
	DocHash  string `json:"doc_hash"`
	ChunkIdx int    `json:"chunk_idx"`
	Content  string `json:"content"`
}

func attachIdentities(hits []hit, rows []snapshotRow) error {
	byID := map[int64]snapshotRow{}
	for _, row := range rows {
		if _, exists := byID[row.PGID]; exists || row.DocHash == "" || row.ChunkIdx < 0 {
			return errors.New("invalid live identity")
		}
		byID[row.PGID] = row
	}
	if len(byID) != len(hits) {
		return errors.New("live identity count mismatch")
	}
	for i, hit := range hits {
		row, ok := byID[hit.PGID]
		if !ok || fmt.Sprintf("%x", sha256.Sum256([]byte(row.Content))) != hit.ContentSHA256 {
			return errors.New("live child changed during retrieval")
		}
		hits[i].DocHash, hits[i].ChunkIdx = row.DocHash, row.ChunkIdx
	}
	return nil
}

func loadIdentities(ctx context.Context, db *sql.DB, hits []hit) error {
	if len(hits) == 0 {
		return nil
	}
	ids := make([]int64, len(hits))
	for i, hit := range hits {
		ids[i] = hit.PGID
	}
	rows, err := db.QueryContext(ctx, "SELECT id, doc_hash, chunk_idx, content FROM rag_chunks WHERE id = ANY($1)", pq.Array(ids))
	if err != nil {
		return err
	}
	defer rows.Close()
	identities := []snapshotRow{}
	for rows.Next() {
		var row snapshotRow
		if err := rows.Scan(&row.PGID, &row.DocHash, &row.ChunkIdx, &row.Content); err != nil {
			return err
		}
		identities = append(identities, row)
	}
	if err := rows.Err(); err != nil {
		return err
	}
	return attachIdentities(hits, identities)
}

func snapshot(ctx context.Context, db *sql.DB) ([]snapshotRow, error) {
	tx, err := db.BeginTx(ctx, &sql.TxOptions{ReadOnly: true, Isolation: sql.LevelRepeatableRead})
	if err != nil {
		return nil, err
	}
	defer tx.Rollback()
	rows, err := tx.QueryContext(ctx, "SELECT id, doc_hash, chunk_idx, content FROM rag_chunks ORDER BY id")
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	result := []snapshotRow{}
	for rows.Next() {
		var row snapshotRow
		if err := rows.Scan(&row.PGID, &row.DocHash, &row.ChunkIdx, &row.Content); err != nil {
			return nil, err
		}
		result = append(result, row)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	if err := tx.Commit(); err != nil {
		return nil, err
	}
	return result, nil
}

func sourceRevision() (string, error) {
	info, ok := debug.ReadBuildInfo()
	if !ok {
		return "", errors.New("build metadata missing")
	}
	var revision string
	for _, s := range info.Settings {
		if s.Key == "vcs.revision" {
			revision = s.Value
		}
		if s.Key == "vcs.modified" && s.Value == "true" {
			return "", errors.New("build has uncommitted changes")
		}
	}
	if revision == "" {
		return "", errors.New("build VCS revision missing; build from a clean Git checkout")
	}
	return revision, nil
}

func run(args []string, input io.Reader, output io.Writer) error {
	f := flag.NewFlagSet("rag-export", flag.ContinueOnError)
	settings := f.String("backend-config", "", "private JSON settings containing existing credential environment variable names")
	read := f.Bool("allow-backend-read", false, "explicitly permit connections to existing read-only evaluation backends")
	models := f.Bool("allow-model-calls", false, "explicitly permit embedding/rerank calls; obtain budget approval separately")
	snap := f.Bool("snapshot", false, "export original PG child rows as JSONL; no model/ES/Milvus access")
	deadline := f.Duration("timeout", 45*time.Second, "operation timeout (outer process deadline also required)")
	if err := f.Parse(args); err != nil {
		return err
	}
	if !*read || (!*snap && !*models) {
		return errors.New("explicit backend/model opt-in required")
	}
	if *settings == "" || *deadline <= 0 || *deadline > 5*time.Minute {
		return errors.New("invalid settings/timeout")
	}
	var req request
	if !*snap {
		dec := json.NewDecoder(io.LimitReader(input, 1<<20))
		dec.DisallowUnknownFields()
		if err := dec.Decode(&req); err != nil {
			return errors.New("invalid request JSON")
		}
		var extra any
		if err := dec.Decode(&extra); err != io.EOF {
			return errors.New("extra request data")
		}
		if err := validateRequest(req); err != nil {
			return err
		}
	}
	data, err := os.ReadFile(*settings)
	if err != nil {
		return errors.New("cannot read backend settings")
	}
	var backend backendConfig
	if err := json.Unmarshal(data, &backend); err != nil {
		return errors.New("invalid backend settings")
	}
	dsn := os.Getenv(backend.PGDSNEnv)
	if dsn == "" {
		return errors.New("existing PG credential environment variable unavailable")
	}
	// Direct database opening avoids platform bootstrap/migrations and DSN logging.
	ctx, cancel := context.WithTimeout(context.Background(), *deadline)
	defer cancel()
	db, err := sql.Open("postgres", dsn)
	if err != nil {
		return errors.New("PG open failed")
	}
	defer db.Close()
	db.SetMaxOpenConns(2)
	db.SetMaxIdleConns(1)
	if *snap {
		rows, err := snapshot(ctx, db)
		if err != nil {
			return errors.New("PG snapshot failed")
		}
		enc := json.NewEncoder(output)
		for _, row := range rows {
			if err := enc.Encode(row); err != nil {
				return err
			}
		}
		return nil
	}
	revision, err := sourceRevision()
	if err != nil {
		return err
	}
	if err := checkExpected(req, backend, revision); err != nil {
		return err
	}
	if backend.EmbeddingModel == "" || backend.EmbeddingURL == "" || os.Getenv(backend.EmbeddingKeyEnv) == "" || backend.MilvusAddress == "" {
		return errors.New("existing embedding/Milvus settings unavailable")
	}
	if req.Mode != "vector" && len(backend.ESAddresses) == 0 {
		return errors.New("ES settings missing")
	}
	if req.Mode == "hybrid_rerank" && (backend.RerankModel == "" || backend.RerankURL == "" || os.Getenv(backend.RerankKeyEnv) == "" || backend.RerankPreviewLen <= 0) {
		return errors.New("existing rerank settings unavailable")
	}
	if err := db.PingContext(ctx); err != nil {
		return errors.New("PG unavailable")
	}
	mc, err := milvus.NewClient(ctx, milvus.Config{Address: backend.MilvusAddress})
	if err != nil {
		return errors.New("Milvus unavailable")
	}
	defer mc.Close()
	var ec *es.Client
	if req.Mode != "vector" {
		ec, err = es.NewClient(es.Config{Addresses: backend.ESAddresses, Username: os.Getenv(backend.ESUsernameEnv), Password: os.Getenv(backend.ESPasswordEnv)})
		if err != nil {
			return errors.New("ES client unavailable")
		}
	}
	cfg := &config.APIConfig{
		LLMConfig:       config.LLMConfig{LLMMode: "real", LLMAPIUrl: backend.RerankURL, LLMAPIKey: os.Getenv(backend.RerankKeyEnv), LLMModel: backend.RerankModel},
		EmbeddingConfig: config.EmbeddingConfig{EmbeddingAPIUrl: backend.EmbeddingURL, EmbeddingAPIKey: os.Getenv(backend.EmbeddingKeyEnv), EmbeddingModel: backend.EmbeddingModel},
		RAGConfig:       config.RAGConfig{ChunkSize: 200, TopK: 5, RRFConstantK: req.RRFK},
	}
	client := llm.New(cfg)
	engine := rag.NewEngine(cfg, ragchunk.NewStore(db, mc, ec), nil)
	engine.SetEmbedFn(client.EmbedContext)
	var reranker any
	if req.Mode == "hybrid_rerank" {
		engine.SetReranker(rag.NewLLMReranker(func(ctx context.Context, system, user string) (string, error) {
			return client.ChatContext(ctx, system, []llm.Message{{Role: "user", Content: user}})
		}, backend.RerankPreviewLen))
		reranker = backend.RerankModel
	}
	started := time.Now()
	hits, err := retrieve(ctx, req, engine)
	if err != nil {
		return errors.New("retrieval failed")
	} // Never serialize SDK/credential-bearing error bodies.
	if err := loadIdentities(ctx, db, hits); err != nil {
		return errors.New("live identity verification failed")
	}
	if ctx.Err() != nil {
		return errors.New("retrieval deadline exceeded")
	}
	return json.NewEncoder(output).Encode(map[string]any{
		"kind": "knowflow_backend_export_v1", "question_sha256": fmt.Sprintf("%x", sha256.Sum256([]byte(req.Question))),
		"mode": req.Mode, "hits": hits, "retrieval_latency_ms": float64(time.Since(started).Microseconds()) / 1000,
		"actual_config": map[string]any{"revision": revision, "embedding_model": backend.EmbeddingModel,
			"reranker": reranker, "rerank_preview_len": backend.RerankPreviewLen, "candidate_k": req.CandidateK,
			"rrf_k": req.RRFK, "top_k": 5, "rewrite": false, "kg": false},
	})
}

func main() {
	// SDK diagnostics may contain response bodies or connection strings. Export only fixed error categories.
	logger.Init(logger.Config{Output: io.Discard})
	log.SetOutput(io.Discard)
	if err := run(os.Args[1:], os.Stdin, os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, "rag-export:", err)
		os.Exit(1)
	}
}
