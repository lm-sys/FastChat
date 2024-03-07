import sqlite3


class ModelResponseCache:
    def __init__(self, db_path='model_responses_cache.db'):
        self.db_path = db_path
        self._initialize_db()

    def _initialize_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
        CREATE TABLE IF NOT EXISTS responses (
            id INTEGER PRIMARY KEY,
            model_name TEXT,
            prompt TEXT,
            output TEXT,
            UNIQUE(model_name, prompt)
        )''')
        conn.commit()
        conn.close()

    def cache_response(self, model_name, prompt, output):
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute('''
            INSERT OR REPLACE INTO responses (model_name, prompt, output)
            VALUES (?, ?, ?)
            ''', (model_name, prompt, output))
            conn.commit()

    def get_cached_response(self, model_name, prompt):
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute('''
            SELECT output FROM responses WHERE model_name = ? AND prompt = ?
            ''', (model_name, prompt))
            result = c.fetchone()
            if result:
                return result[0]

    def is_key_stored(self, model_name, prompt):
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute('''
            SELECT EXISTS(SELECT 1 FROM responses WHERE model_name = ? AND prompt = ? LIMIT 1)
            ''', (model_name, prompt))
            return c.fetchone()[0] == 1