import sqlite3
c = sqlite3.connect("data/verification/authorities.db")
c.row_factory = sqlite3.Row
rows = c.execute("SELECT act, act_norm, kind, number FROM provisions WHERE act_norm LIKE '%pmla%' OR act_norm LIKE '%prevention of money laundering%' LIMIT 10").fetchall()
for r in rows:
    print(dict(r))
