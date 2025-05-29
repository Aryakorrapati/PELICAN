import sqlite3
conn = sqlite3.connect('/eos/user/a/akorrapa/test_study.db')
c = conn.cursor()
c.execute('CREATE TABLE test (id INTEGER PRIMARY KEY)')
conn.commit()
conn.close()
