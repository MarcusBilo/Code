import tkinter as tk
import sqlite3


def connect_to_db():
    conn = sqlite3.connect('my_database.db')
    return conn


def execute_query(query, parameters=None):
    conn = connect_to_db()
    cursor = conn.cursor()
    if parameters:
        cursor.execute(query, parameters)
    else:
        cursor.execute(query)
    conn.commit()
    conn.close()


def create_animal_window():

    def create_animal():
        genus_value = entry_genus.get()
        insert_query = "INSERT INTO Animal (Genus) VALUES (?)"
        execute_query(insert_query, (genus_value,))
        load_animals()
        entry_id.config(state='normal')
        entry_id.delete(0, 'end')
        entry_id.config(state='readonly')
        entry_genus.delete(0, 'end')
        load_animals()

    def delete_animal():
        selected_animal = option_var.get()
        if selected_animal:
            animal_id = int(selected_animal.split()[0])
            delete_query = "DELETE FROM Animal WHERE ID = ?"
            execute_query(delete_query, (animal_id,))
            load_animals()
            entry_id.config(state='normal')
            entry_id.delete(0, 'end')
            entry_id.config(state='readonly')
            entry_genus.delete(0, 'end')

    def load_animals():
        conn = sqlite3.connect('my_database.db')
        cursor = conn.cursor()
        cursor.execute("SELECT ID, Genus FROM Animal")
        animals = cursor.fetchall()
        animal_options = [f"{animal[0]} - {animal[1]}" for animal in animals]
        option_var.set("")
        option_menu['menu'].delete(0, 'end')
        for option in animal_options:
            option_menu['menu'].add_command(label=option, command=lambda value=option: option_var.set(value))
        conn.close()

    def select_animal():
        selected_animal = option_var.get()
        if selected_animal:
            animal_id = int(selected_animal.split()[0])
            conn = sqlite3.connect('my_database.db')
            cursor = conn.cursor()
            cursor.execute("SELECT Genus FROM Animal WHERE ID=?", (animal_id,))
            selected_genus = cursor.fetchone()
            if selected_genus:
                entry_id.config(state='normal')
                entry_id.delete(0, 'end')
                entry_id.insert(0, str(animal_id))
                entry_id.config(state='readonly')
                entry_genus.delete(0, 'end')
                entry_genus.insert(0, selected_genus[0])
            conn.close()

    new_window = tk.Toplevel(root)
    new_window.title("Create Animal")

    tk.Label(new_window, text="ID").grid(row=0, column=0, pady=5, padx=5, sticky="e")
    entry_id = tk.Entry(new_window, state="readonly")
    entry_id.grid(row=0, column=1, pady=5, padx=5, sticky="w")

    tk.Label(new_window, text="Genus").grid(row=0, column=2, pady=5, padx=5, sticky="e")
    entry_genus = tk.Entry(new_window)
    entry_genus.grid(row=0, column=3, pady=5, padx=5, sticky="w")

    tk.Label(new_window, text="Select Animal").grid(row=0, column=4, pady=5, padx=5, sticky="e")
    option_var = tk.StringVar(new_window)
    option_menu = tk.OptionMenu(new_window, option_var, "")
    option_menu.grid(row=0, column=5, pady=5, padx=5, sticky="w")

    create_animal_button = tk.Button(new_window, text="Create Animal", command=create_animal)
    create_animal_button.grid(row=1, column=0, columnspan=2, pady=10, sticky="nsew")

    select_animal_button = tk.Button(new_window, text="Load Animal", command=select_animal)
    select_animal_button.grid(row=1, column=2, columnspan=2, pady=10, sticky="nsew")

    delete_animal_button = tk.Button(new_window, text="Delete Animal", command=delete_animal)
    delete_animal_button.grid(row=1, column=4, columnspan=2, pady=10, sticky="nsew")

    load_animals()


def create_location_window():
    new_window = tk.Toplevel(root)
    new_window.title("Create Location")
    tk.Label(new_window, text="LNr").grid(row=1, column=0, pady=5, padx=5, sticky="e")
    tk.Entry(new_window, state="readonly").grid(row=1, column=1, pady=5, padx=5, sticky="w")
    tk.Label(new_window, text="Shorttitle").grid(row=2, column=0, pady=5, padx=5, sticky="e")
    tk.Entry(new_window).grid(row=2, column=1, pady=5, padx=5, sticky="w")
    tk.Label(new_window, text="Description").grid(row=3, column=0, pady=5, padx=5, sticky="e")
    tk.Entry(new_window).grid(row=3, column=1, pady=5, padx=5, sticky="w")
    tk.Button(new_window, text="Create Location").grid(row=4, column=0, columnspan=2, pady=10, sticky="nsew")
    tk.Label(new_window, text="LNr").grid(row=1, column=2, pady=5, padx=5, sticky="e")
    tk.OptionMenu(new_window, tk.StringVar(new_window), "Option 1", "Option 2", "Option 3").grid(row=1, column=3, pady=5, padx=5, sticky="w")
    tk.Label(new_window, text="").grid(row=2, column=2, rowspan=2)  # Invisible labels
    tk.Button(new_window, text="Delete Location").grid(row=4, column=2, columnspan=2, pady=10, sticky="nsew")


def enter_observation_window():
    new_window = tk.Toplevel(root)
    new_window.title("Enter Observation")
    tk.Label(new_window, text="Animal").grid(row=1, column=0, pady=5, padx=5, sticky="e")
    tk.OptionMenu(new_window, tk.StringVar(new_window), "Option 1", "Option 2", "Option 3").grid(row=1, column=1, pady=5, padx=5, sticky="w")
    tk.Label(new_window, text="Gender").grid(row=2, column=0, pady=5, padx=5, sticky="e")
    tk.OptionMenu(new_window, tk.StringVar(new_window), "Option 1", "Option 2", "Option 3").grid(row=2, column=1, pady=5, padx=5, sticky="w")
    tk.Label(new_window, text="Date").grid(row=3, column=0, pady=5, padx=5, sticky="e")
    tk.Entry(new_window).grid(row=3, column=1, pady=5, padx=5, sticky="w")
    tk.Label(new_window, text="Time").grid(row=4, column=0, pady=5, padx=5, sticky="e")
    tk.Entry(new_window).grid(row=4, column=1, pady=5, padx=5, sticky="w")
    tk.Label(new_window, text="Location").grid(row=1, column=2, pady=5, padx=5, sticky="e")
    tk.OptionMenu(new_window, tk.StringVar(new_window), "Option 1", "Option 2", "Option 3").grid(row=1, column=3, pady=5, padx=5, sticky="w")
    tk.Label(new_window, text="Estimated Age").grid(row=2, column=2, pady=5, padx=5, sticky="e")
    tk.Entry(new_window).grid(row=2, column=3, pady=5, padx=5, sticky="w")
    tk.Label(new_window, text="Estimated Weight").grid(row=3, column=2, pady=5, padx=5, sticky="e")
    tk.Entry(new_window).grid(row=3, column=3, pady=5, padx=5, sticky="w")
    tk.Label(new_window, text="Estimated Size").grid(row=4, column=2, pady=5, padx=5, sticky="e")
    tk.Entry(new_window).grid(row=4, column=3, pady=5, padx=5, sticky="w")
    tk.Button(new_window, text="Clear").grid(row=5, column=0, columnspan=2, pady=10, sticky="nsew")
    tk.Button(new_window, text="Save").grid(row=5, column=2, columnspan=2, pady=10, sticky="nsew")


def exit_application():
    root.destroy()


def main():
    create_table_query = "CREATE TABLE IF NOT EXISTS Animal (id INTEGER PRIMARY KEY AUTOINCREMENT, Genus TEXT)"
    sqlite3.connect('my_database.db').cursor().execute(create_table_query)

    global root
    root = tk.Tk()
    root.title("Wildlife Management")
    root.geometry("444x250")  # Width x Height

    button1 = tk.Button(root, text="Create Animal", command=create_animal_window)
    button2 = tk.Button(root, text="Create Location", command=create_location_window)
    button3 = tk.Button(root, text="Enter Observation", command=enter_observation_window)
    button4 = tk.Button(root, text="Exit", command=exit_application)

    button1.grid(row=1, column=0, pady=15, sticky="nsew")
    button2.grid(row=2, column=0, pady=15, sticky="nsew")
    button3.grid(row=3, column=0, pady=15, sticky="nsew")
    button4.grid(row=4, column=0, pady=15, sticky="nsew")

    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(0, weight=1)

    root.mainloop()


if __name__ == "__main__":
    main()
