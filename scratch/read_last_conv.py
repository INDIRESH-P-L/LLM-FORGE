from storage import chat_store

from storage.chat_store import get_connection, get_messages

conn = get_connection()
rows = conn.execute("SELECT * FROM conversations ORDER BY updated_at DESC LIMIT 3").fetchall()
print(f"Total conversations in DB: {len(rows)}")
for r in rows:
    print(f"Conv ID: {r['id']} | Title: {r['title']}")
    msgs = get_messages(r['id'])
    print(f"Messages count: {len(msgs)}")
    for m in msgs:
        print(f"  [{m.role.upper()}] Status: {m.status} | Latency: {m.latency_ms}ms | Content length: {len(m.content)}")
        print(f"    Citations count: {len(m.citations)}")
        if m.citations:
            for c in m.citations[:4]:
                print(f"      - {c.get('case_name')} | {c.get('citation')}")
