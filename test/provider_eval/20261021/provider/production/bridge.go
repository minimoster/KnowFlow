package rag

import (
 "context"
 "crypto/sha256"
 "encoding/hex"
 "encoding/json"
 "errors"
 "fmt"
 "math"
 "unicode/utf8"
)

// The production reranker.go is unchanged. Only dependency types are supplied.
type Chunk struct { Content string `json:"content"` }
type HybridResult struct {
 ID string `json:"id"`
 Chunk Chunk `json:"chunk"`
 Score float64 `json:"score"`
 Source string `json:"source"`
}
func runeLen(s string) int { return utf8.RuneCountInString(s) }
type Input struct {
 Mode string `json:"mode"`
 Query string `json:"query"`
 Candidates []HybridResult `json:"candidates"`
 Raw string `json:"raw"`
 Canceled bool `json:"canceled"`
 GeneratorError bool `json:"generator_error"`
}
type Output struct {
 Results []HybridResult `json:"results,omitempty"`
 System string `json:"system,omitempty"`
 User string `json:"user,omitempty"`
 TemplateSHA256 string `json:"template_sha256"`
 Error string `json:"error,omitempty"`
 Calls int `json:"callback_calls"`
}
func Evaluate(in Input) Output {
 h:=sha256.Sum256([]byte(rerankSystemPrompt)); out:=Output{TemplateSHA256:hex.EncodeToString(h[:])}
 if len(in.Candidates)!=20 || in.Query=="" { out.Error="requires one query and 20 frozen candidates"; return out }
 ids:=map[string]bool{}
 for _,c:=range in.Candidates {
  if c.ID=="" || ids[c.ID] || c.Chunk.Content=="" || math.IsNaN(c.Score) || math.IsInf(c.Score,0) {out.Error="invalid candidate identity/content/score";return out}
  ids[c.ID]=true
 }
 if in.Mode!="prompt" && in.Mode!="parse" {out.Error="invalid mode";return out}
 raw:=in.Raw
 if in.Mode=="prompt" {
  a:=make([]map[string]int,20);for i:=range a {a[i]=map[string]int{"idx":i,"score":0}}
  b,_:=json.Marshal(map[string]any{"scores":a});raw=string(b)
 }
 ctx,cancel:=context.WithCancel(context.Background());defer cancel();if in.Canceled {cancel()}
 engine:=NewLLMReranker(func(ctx context.Context,system,user string)(string,error){
  out.Calls++;out.System=system;out.User=user
  if in.GeneratorError{return "",errors.New("offline generator failure")}
  return raw,nil
 },200)
 result,err:=engine.Rerank(ctx,in.Query,in.Candidates,5)
 if err!=nil {out.Error=fmt.Sprint(err);return out}
 if in.Mode=="parse" {out.Results=result}
 return out
}
