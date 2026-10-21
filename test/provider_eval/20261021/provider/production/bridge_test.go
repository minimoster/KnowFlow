package rag

import (
 "encoding/json"
 "fmt"
 "strings"
 "testing"
)
func fixture() Input {
 a:=make([]HybridResult,20)
 for i:=range a {a[i]=HybridResult{ID:fmt.Sprint(i),Chunk:Chunk{Content:strings.Repeat("中",205)},Score:1/float64(i+1),Source:"hybrid"}}
 return Input{Mode:"parse",Query:"synthetic query",Candidates:a}
}
func scores(change func([]map[string]any)) string {
 a:=make([]map[string]any,20);for i:=range a {a[i]=map[string]any{"idx":i,"score":5}}
 if change!=nil {change(a)};b,_:=json.Marshal(map[string]any{"scores":a});return string(b)
}
func TestProductionPromptAndRuneBoundary(t *testing.T){
 in:=fixture();in.Mode="prompt";out:=Evaluate(in)
 if out.Error!="" || out.Calls!=1 || !strings.Contains(out.User,strings.Repeat("中",200)+"…") || strings.Contains(out.User,strings.Repeat("中",201)){t.Fatal(out.Error)}
 if out.System!=rerankSystemPrompt {t.Fatal("production prompt changed")}
}
func TestProductionTieBehavior(t *testing.T){
 in:=fixture();in.Raw=scores(nil);out:=Evaluate(in)
 if out.Error!="" || len(out.Results)!=5 {t.Fatal(out.Error)}
 for i,c:=range out.Results {if c.ID!=fmt.Sprint(i) || c.Score!=.5 || c.Source!="hybrid+rerank"{t.Fatal(c)}}
 in.Candidates[0].Score=in.Candidates[1].Score;out=Evaluate(in)
 if out.Results[0].ID!="0"{t.Fatal("stable ties changed")}
}
func TestProductionRejectsMalformedScores(t *testing.T){
 cases:=map[string]string{"bad_json":"not json","missing":"{\"scores\":[]}","nan":"{\"scores\":[{\"idx\":0,\"score\":NaN}]}","duplicate":scores(func(a []map[string]any){a[1]["idx"]=0}),"range":scores(func(a []map[string]any){a[1]["score"]=11}),"fraction":scores(func(a []map[string]any){a[1]["score"]=1.5}),"index":scores(func(a []map[string]any){a[1]["idx"]=-1})}
 for name,raw:=range cases {t.Run(name,func(t *testing.T){in:=fixture();in.Raw=raw;if Evaluate(in).Error==""{t.Fatal("accepted")}})}
}
func TestProductionFenceCancellationAndGeneratorError(t *testing.T){
 in:=fixture();in.Raw="```json\n"+scores(nil)+"\n```";if Evaluate(in).Error!=""{t.Fatal("production fence support changed")}
 in.Canceled=true;out:=Evaluate(in);if out.Error=="" || out.Calls!=0{t.Fatal("cancel dispatched")}
 in.Canceled=false;in.GeneratorError=true;out=Evaluate(in);if out.Error=="" || len(out.Results)!=0{t.Fatal("silent fallback")}
}
