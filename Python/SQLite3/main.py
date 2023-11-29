import sqlite3

conn = sqlite3.connect('my_database.db')

create_table_query = "CREATE TABLE IF NOT EXISTS example_table (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, age INTEGER, email TEXT)"
conn.cursor().execute(create_table_query)

insert_query = "INSERT INTO example_table (name, age, email) VALUES ('Alice', 25, 'alice@example.com'), ('Bob', 30, 'bob@example.com')"

conn.execute(insert_query)
conn.commit()

with open("dump.sql", "w") as f:
    for line in conn.iterdump():
        f.write("%s\n" % line)

conn.close()
