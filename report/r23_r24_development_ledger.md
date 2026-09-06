# R23/R24 pre-freeze development ledger

- Before any contract was frozen or GPU run launched, the first summarizer
  self-check exited 1 because its toy fixture encoded the intended
  reported-versus-exact diagonal-trace delta as the decimal literal `1e-7`
  rather than recomputing the representable subtraction.  The summarizer
  correctly rejected that inconsistent fixture.  The fixture now derives
  the delta from the stored toy values; both runner and summarizer
  self-checks subsequently exit 0.  No experimental output was produced.
