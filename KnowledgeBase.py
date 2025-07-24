import threading

class KB():
    def __init__(self):
        self.db = {}
        self.lock = threading.Lock()
        return
    
    def store_var(self, var_name, data):
        with self.lock:
            self.db[var_name] = data
        print("Updated variable", var_name, "to ", data)
        return
    
    def get_var(self, var_name):
        with self.lock:
            if var_name in self.db.keys():
                return self.db[var_name]
            else:
                print("Variable ", var_name, "does not exist in database.")
                return None
        
    def get_db(self):
        with self.lock:
            if len(self.db)==0:
                print("Database is empty")
                return
            else:
                return self.db
    
    def clear_db(self):
        with self.lock:
            self.db = {}
            print("Completely cleared database.")
            return