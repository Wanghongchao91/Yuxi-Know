from src.knowledge.implementations.lightrag import LightRagKB
from neo4j._async.work import transaction as _txn_mod

class DummyTx:
    pass

def stub_run(self, query, parameters=None, *args, **kwargs):
    print(query)
    return query

_txn_mod.Transaction.run = stub_run
kb = LightRagKB(work_dir='.')
kb._patch_neo4j_async_transaction_run()
q1 = "MATCH (n) SET n:``artifact`"
q2 = "SET n:`ws-1`:artifact:``foo-bar``"
_txn_mod.Transaction.run(DummyTx(), q1)
_txn_mod.Transaction.run(DummyTx(), q2)
