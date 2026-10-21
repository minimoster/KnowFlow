package main

import (
 "bufio"
 "encoding/json"
 "os"
 rag "knowflow-prepare/production"
)
func main(){
 s:=bufio.NewScanner(os.Stdin);s.Buffer(make([]byte,4096),1<<20)
 out:=json.NewEncoder(os.Stdout)
 for s.Scan(){
  var in rag.Input
  if err:=json.Unmarshal(s.Bytes(),&in);err!=nil {out.Encode(map[string]string{"error":"invalid input JSON"});continue}
  out.Encode(rag.Evaluate(in))
 }
 if s.Err()!=nil {os.Exit(2)}
}
