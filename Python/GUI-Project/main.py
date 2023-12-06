import tkinter as tk
import sqlite3


def main():
    create_animal_table = "CREATE TABLE IF NOT EXISTS Animal (id INTEGER PRIMARY KEY AUTOINCREMENT, Genus TEXT)"
    create_location_table = "CREATE TABLE IF NOT EXISTS Location (LNr INTEGER PRIMARY KEY AUTOINCREMENT, Shorttitle TEXT, Description TEXT)"

    execute_query(create_animal_table)
    execute_query(create_location_table)

    global root
    root = tk.Tk()
    root.title("Wildlife Management")
    root.geometry("444x250")

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


def execute_query(query, parameters=None):
    conn = connect_to_db()
    cursor = conn.cursor()
    if parameters:
        cursor.execute(query, parameters)
    else:
        cursor.execute(query)
    conn.commit()
    conn.close()


def connect_to_db():
    return sqlite3.connect('my_database.db')


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
        conn = connect_to_db()
        cursor = conn.cursor()
        cursor.execute("SELECT ID, Genus FROM Animal")
        animals = cursor.fetchall()
        animal_options = [f"{animal[0]} - {animal[1]}" for animal in animals]
        option_var.set("")
        option_menu['menu'].delete(0, 'end')
        for option in animal_options:
            command = lambda value=option: option_var.set(value)
            option_menu['menu'].add_command(label=option, command=command)
        conn.close()

    def select_animal():
        selected_animal = option_var.get()
        if selected_animal:
            animal_id = int(selected_animal.split()[0])
            conn = connect_to_db()
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

    label_id = tk.Label(new_window, text="ID")
    label_id.grid(row=0, column=0, pady=5, padx=5, sticky="e")
    entry_id = tk.Entry(new_window, state="readonly")
    entry_id.grid(row=0, column=1, pady=5, padx=5, sticky="w")

    label_genus = tk.Label(new_window, text="Genus")
    label_genus.grid(row=0, column=2, pady=5, padx=5, sticky="e")
    entry_genus = tk.Entry(new_window)
    entry_genus.grid(row=0, column=3, pady=5, padx=5, sticky="w")

    label_select = tk.Label(new_window, text="Select Animal")
    label_select.grid(row=0, column=4, pady=5, padx=5, sticky="e")
    option_var = tk.StringVar(new_window)
    option_menu = tk.OptionMenu(new_window, option_var, "")
    option_menu.grid(row=0, column=5, pady=5, padx=5, sticky="w")

    btn_create_animal = tk.Button(new_window, text="Create Animal", command=create_animal)
    btn_create_animal.grid(row=1, column=0, columnspan=2, pady=10, sticky="nsew")

    btn_select_animal = tk.Button(new_window, text="Load Animal", command=select_animal)
    btn_select_animal.grid(row=1, column=2, columnspan=2, pady=10, sticky="nsew")

    btn_delete_animal = tk.Button(new_window, text="Delete Animal", command=delete_animal)
    btn_delete_animal.grid(row=1, column=4, columnspan=2, pady=10, sticky="nsew")

    load_animals()


def create_location_window():

    def create_location():
        short_title_value = entry_shorttitel.get()
        description_value = entry_description.get()
        insert_query = "INSERT INTO Location (Shorttitle, Description) VALUES (?, ?)"
        execute_query(insert_query, (short_title_value, description_value))
        load_locations()
        entry_lnr.config(state='normal')
        entry_lnr.delete(0, 'end')
        entry_lnr.config(state='readonly')
        entry_shorttitel.delete(0, 'end')
        entry_description.delete(0, 'end')

    def delete_location():
        selected_location = option_var.get()
        if selected_location:
            location_id = int(selected_location.split()[0])
            delete_query = "DELETE FROM Location WHERE LNr = ?"
            execute_query(delete_query, (location_id,))
            load_locations()
            entry_lnr.config(state='normal')
            entry_lnr.delete(0, 'end')
            entry_lnr.config(state='readonly')
            entry_shorttitel.delete(0, 'end')
            entry_description.delete(0, 'end')

    def load_locations():
        conn = connect_to_db()
        cursor = conn.cursor()
        cursor.execute("SELECT LNr, Shorttitle, Description FROM Location")
        locations = cursor.fetchall()
        location_options = [f"{location[0]} - {location[1]} - {location[2]}" for location in locations]
        option_var.set("")
        option_menu['menu'].delete(0, 'end')
        for option in location_options:
            command = lambda value=option: option_var.set(value)
            option_menu['menu'].add_command(label=option, command=command)
        conn.close()

    def select_location():
        selected_location = option_var.get()
        if selected_location:
            location_id = int(selected_location.split()[0])
            conn = connect_to_db()
            cursor = conn.cursor()
            cursor.execute("SELECT LNr, Shorttitle, Description FROM Location WHERE LNr=?", (location_id,))
            selected_location = cursor.fetchone()
            if selected_location:
                entry_lnr.config(state='normal')
                entry_lnr.delete(0, 'end')
                entry_lnr.insert(0, str(selected_location[0]))
                entry_lnr.config(state='readonly')
                entry_shorttitel.delete(0, 'end')
                entry_shorttitel.insert(0, selected_location[1])
                entry_description.delete(0, 'end')
                entry_description.insert(0, selected_location[2])
            conn.close()

    new_window = tk.Toplevel(root)
    new_window.title("Create Location")

    label_lnr = tk.Label(new_window, text="LNr")
    label_lnr.grid(row=0, column=0, pady=5, padx=5, sticky="e")
    entry_lnr = tk.Entry(new_window, state="readonly")
    entry_lnr.grid(row=0, column=1, pady=5, padx=5, sticky="w")

    label_shorttitel = tk.Label(new_window, text="Shorttitle")
    label_shorttitel.grid(row=0, column=2, pady=5, padx=5, sticky="e")
    entry_shorttitel = tk.Entry(new_window)
    entry_shorttitel.grid(row=0, column=3, pady=5, padx=5, sticky="w")

    label_description = tk.Label(new_window, text="Description")
    label_description.grid(row=0, column=4, pady=5, padx=5, sticky="e")
    entry_description = tk.Entry(new_window)
    entry_description.grid(row=0, column=5, pady=5, padx=5, sticky="w")

    label_select = tk.Label(new_window, text="Select Location")
    label_select.grid(row=0, column=6, pady=5, padx=5, sticky="e")
    option_var = tk.StringVar(new_window)
    option_menu = tk.OptionMenu(new_window, option_var, "")
    option_menu.grid(row=0, column=7, pady=5, padx=5, sticky="w")

    btn_create_location = tk.Button(new_window, text="Create Location", command=create_location)
    btn_create_location.grid(row=1, column=0, columnspan=3, pady=10, padx=5, sticky="nsew")

    btn_select_location = tk.Button(new_window, text="Load Location", command=select_location)
    btn_select_location.grid(row=1, column=3, columnspan=3, pady=10, padx=5, sticky="nsew")

    btn_delete_location = tk.Button(new_window, text="Delete Location", command=delete_location)
    btn_delete_location.grid(row=1, column=6, columnspan=3, pady=10, padx=5, sticky="nsew")

    load_locations()


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


if __name__ == "__main__":
    main()
