#!/usr/bin/env python3
"""Run the same SYNTHETIC failure contract against immutable git revisions.

This is reliability regression evidence, not RAG accuracy/latency/model evaluation.
No network transport, API credentials, GPU, services or Docker are used.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
import tempfile

PROBE = r'''package llm
import (
 "context"
 "io"
 "net/http"
 "strings"
 "testing"
 "time"
 "agi-assistant/config"
)
type comparisonTransport func(*http.Request)(*http.Response,error)
func(f comparisonTransport) RoundTrip(r *http.Request)(*http.Response,error){return f(r)}
func comparisonResponse(code int, body string)*http.Response{
 return &http.Response{StatusCode:code,Header:make(http.Header),Body:io.NopCloser(strings.NewReader(body))}
}
func comparisonClient(f comparisonTransport)*Client{
 c:=New(&config.APIConfig{LLMConfig:config.LLMConfig{LLMAPIUrl:"http://synthetic.invalid",LLMAPIKey:"synthetic-not-a-key",LLMModel:"synthetic"}})
 c.httpClient=&http.Client{Transport:f};return c
}
// __CALL_ADAPTER__
func TestComparisonContract(t *testing.T){
 for _,name:=range []string{"HTTP429","HTTP503","ClientTimeout","TaskCanceledInFlight","EmptyResponse","MalformedJSON","TruncatedStream","LateCancellation","MissingConfiguration"}{
  t.Run(name,func(t *testing.T){
   ctx,cancel:=context.WithCancel(context.Background());defer cancel()
   c:=comparisonClient(func(r *http.Request)(*http.Response,error){
    switch name{
    case "HTTP429":return comparisonResponse(429,`{"error":{"message":"synthetic rate limit"}}`),nil
    case "HTTP503":return comparisonResponse(503,`{"error":{"message":"synthetic unavailable"}}`),nil
    case "ClientTimeout":<-r.Context().Done();return nil,r.Context().Err()
    case "TaskCanceledInFlight":cancel();<-r.Context().Done();return nil,r.Context().Err()
    case "EmptyResponse":return comparisonResponse(200,`{"choices":[]}`),nil
    case "MalformedJSON":return comparisonResponse(200,`invalid JSON`),nil
    case "TruncatedStream":return comparisonResponse(200,"data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n"),nil
    case "LateCancellation":cancel();return comparisonResponse(200,`{"choices":[{"message":{"content":"late success"}}]}`),nil
    default:return nil,context.Canceled
    }
   })
   if name=="ClientTimeout"{c.httpClient.Timeout=20*time.Millisecond}
   if name=="MissingConfiguration"{c=New(&config.APIConfig{})}
   out,err:=comparisonCall(c,ctx,name=="TruncatedStream")
   if err==nil || out!=""{t.Fatalf("failure contract violated: nonempty_answer=%t explicit_error=%t",out!="",err!=nil)}
  })
 }
}
'''

CASES = ("HTTP429", "HTTP503", "ClientTimeout", "TaskCanceledInFlight", "EmptyResponse", "MalformedJSON", "TruncatedStream", "LateCancellation", "MissingConfiguration")


def probe_source(legacy):
    normal = 'return c.ChatContext(ctx,"",nil)'
    stream = 'return c.ChatStreamContext(ctx,"",nil,nil)'
    if legacy:
        normal = 'return c.ChatContext(ctx,"",nil),nil'
        stream = 'return c.ChatStreamContext(ctx,"",nil,nil),nil'
    adapter = f'func comparisonCall(c *Client,ctx context.Context,stream bool)(string,error){{if stream{{{stream}}};{normal}}}'
    return PROBE.replace("// __CALL_ADAPTER__", adapter)


def run(root, revisions, output, scratch=None):
    root = Path(root).resolve(); output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    scratch = Path(scratch or root / ".tmp"); scratch.mkdir(parents=True,exist_ok=True)
    env = dict(os.environ, GOMAXPROCS="2", GOTOOLCHAIN="local", GOPROXY="off", GOSUMDB="off")
    report = {"kind":"SYNTHETIC_FAILURE_CONTRACT_COMPARISON_NOT_RAG_QUALITY",
              "contract":"a failed/canceled model call returns empty final answer and explicit error",
              "scope":"in-memory HTTP transport; no tokens, paid calls or model-quality claims",
              "go_version":subprocess.check_output(["go","version"],env=env,text=True).strip(),
              "probe_template_sha256":hashlib.sha256(PROBE.encode()).hexdigest(), "revisions":[]}
    for ref in revisions:
        sha = subprocess.check_output(["git","rev-parse",ref],cwd=root,text=True).strip()
        archive = subprocess.check_output(["git","archive","--format=tar",sha],cwd=root)
        with tempfile.TemporaryDirectory(prefix="knowflow-contract-",dir=scratch) as temp:
            checkout = Path(temp)
            with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
                # Only regular files/directories from a git archive; never links.
                # Avoid tarfile's Windows realpath API, unavailable in this sandbox.
                for member in tar:
                    parts=PurePosixPath(member.name).parts
                    if member.name.startswith("/") or ".." in parts or any(":" in x or "\\" in x for x in parts):
                        raise ValueError("unsafe archive path")
                    target=checkout.joinpath(*parts)
                    if not target.is_relative_to(checkout): raise ValueError("archive escapes checkout")
                    if member.isdir():target.mkdir(parents=True,exist_ok=True)
                    elif member.isfile():
                        target.parent.mkdir(parents=True,exist_ok=True)
                        with tar.extractfile(member) as stream:target.write_bytes(stream.read())
                    else:raise ValueError("archive links/special files are not supported")
            source = (checkout / "internal/infrastructure/llm/llm.go").read_text(encoding="utf-8")
            legacy = "func (c *Client) ChatContext(ctx context.Context, systemPrompt string, messages []Message) string" in source
            probe = probe_source(legacy)
            (checkout / "internal/infrastructure/llm/comparison_probe_test.go").write_text(probe,encoding="utf-8",newline="\n")
            temp_files=checkout / ".comparison-tmp";temp_files.mkdir()
            scoped = dict(env,TEMP=str(temp_files),TMP=str(temp_files),TMPDIR=str(temp_files))
            command=["go","test","-json","-p","1","-count=1","-timeout","15s","./internal/infrastructure/llm","-run","^TestComparisonContract$"]
            completed = subprocess.run(command,cwd=checkout,env=scoped,capture_output=True,text=True,encoding="utf-8",timeout=180)
            (output / f"{sha}.go-test.jsonl").write_text(completed.stdout,encoding="utf-8",newline="\n")
            (output / f"{sha}.stderr.txt").write_text(completed.stderr,encoding="utf-8",newline="\n")
            outcomes={}
            for line in completed.stdout.splitlines():
                try:event=json.loads(line)
                except ValueError:continue
                test=event.get("Test","")
                if test.startswith("TestComparisonContract/") and event.get("Action") in ("pass","fail"):
                    outcomes[test.split("/",1)[1]]=event["Action"]
            if set(outcomes)!=set(CASES):
                raise RuntimeError(f"incomplete probe at {sha}; inspect raw logs (build or timeout failure)")
            if completed.returncode not in (0,1) or (completed.returncode==0)!=all(x=="pass" for x in outcomes.values()):
                raise RuntimeError("process result does not match case outcomes")
            report["revisions"].append({"revision":sha,"legacy_string_only_api":legacy,
                "call_adapter_sha256":hashlib.sha256(probe.encode()).hexdigest(),
                "command":command,"exit_code":completed.returncode,"cases":outcomes,
                "passed":sum(x=="pass" for x in outcomes.values()),"total":len(CASES)})
    (output / "report.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8",newline="\n")
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--revisions",nargs="+",required=True);p.add_argument("--output-dir",required=True)
    p.add_argument("--scratch");a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    result=run(root,a.revisions,a.output_dir,a.scratch)
    print(json.dumps({"kind":result["kind"],"outcomes":[{"revision":r["revision"],"passed":r["passed"],"total":r["total"]} for r in result["revisions"]]}))
