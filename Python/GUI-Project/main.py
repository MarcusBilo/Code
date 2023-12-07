import tkinter as tk
import sqlite3
from datetime import datetime
from tkinter import messagebox
from tkcalendar import DateEntry


def main():
    create_animal_table = "CREATE TABLE IF NOT EXISTS Animal (ID INTEGER PRIMARY KEY AUTOINCREMENT, Genus TEXT)"
    create_location_table = "CREATE TABLE IF NOT EXISTS Location (ID INTEGER PRIMARY KEY AUTOINCREMENT, Shorttitle TEXT, Description TEXT)"
    create_observation_table = """
    CREATE TABLE IF NOT EXISTS Observation (
        ID INTEGER PRIMARY KEY AUTOINCREMENT,
        AnimalID INTEGER,
        LocationID INTEGER,
        Date DATE,
        Time TIME,
        Gender TEXT,
        Age TEXT,
        Weight TEXT,
        Size TEXT,
        FOREIGN KEY (AnimalID) REFERENCES Animal(ID),
        FOREIGN KEY (LocationID) REFERENCES Location(ID)
    )
    """

    execute_query(create_animal_table)
    execute_query(create_location_table)
    execute_query(create_observation_table)

    global root
    root = tk.Tk()
    root.title("Wildlife Management")
    root.resizable(False, False)
    root.geometry("480x270")
    root.configure(bg='lightgray')

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


def execute_query_select(query, parameters=None):
    conn = connect_to_db()
    cursor = conn.cursor()
    if parameters:
        cursor.execute(query, parameters)
    else:
        cursor.execute(query)
    data = cursor.fetchall()
    conn.close()
    return data


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
        query = "SELECT ID, Genus FROM Animal"
        animals = execute_query_select(query)
        animal_options = [f"{animal[0]}" for animal in animals]
        option_var.set("")
        option_menu['menu'].delete(0, 'end')
        for option in animal_options:
            command = lambda value=option: option_var.set(value)
            option_menu['menu'].add_command(label=option, command=command)

    def select_animal():
        selected_animal = option_var.get()
        if selected_animal:
            animal_id = int(selected_animal.split()[0])
            query = "SELECT Genus FROM Animal WHERE ID=?"
            selected_genus = execute_query_select(query, (animal_id,))
            if selected_genus:
                entry_id.config(state='normal')
                entry_id.delete(0, 'end')
                entry_id.insert(0, str(animal_id))
                entry_id.config(state='readonly')
                entry_genus.delete(0, 'end')
                entry_genus.insert(0, selected_genus[0])

    new_window = tk.Toplevel(root)
    new_window.title("Create Animal")

    label_select = tk.Label(new_window, text="Select Animal")
    label_select.grid(row=0, column=0, pady=5, padx=5, sticky="e")
    option_var = tk.StringVar(new_window)
    option_menu = tk.OptionMenu(new_window, option_var, "")
    option_menu.grid(row=0, column=1, pady=5, padx=5, sticky="w")

    label_genus = tk.Label(new_window, text="Genus")
    label_genus.grid(row=0, column=2, pady=5, padx=5, sticky="e")
    entry_genus = tk.Entry(new_window)
    entry_genus.grid(row=0, column=3, pady=5, padx=5, sticky="w")

    label_id = tk.Label(new_window, text="ID")
    label_id.grid(row=0, column=4, pady=5, padx=5, sticky="e")
    entry_id = tk.Entry(new_window, state="readonly")
    entry_id.grid(row=0, column=5, pady=5, padx=5, sticky="w")

    btn_create_animal = tk.Button(new_window, text="Create Animal", command=create_animal)
    btn_create_animal.grid(row=1, column=0, columnspan=2, pady=10, padx=5, sticky="nsew")

    btn_select_animal = tk.Button(new_window, text="Load Animal", command=select_animal)
    btn_select_animal.grid(row=1, column=2, columnspan=2, pady=10, padx=5, sticky="nsew")

    btn_delete_animal = tk.Button(new_window, text="Delete Animal", command=delete_animal)
    btn_delete_animal.grid(row=1, column=4, columnspan=2, pady=10, padx=5, sticky="nsew")

    load_animals()


def create_location_window():

    def create_location():
        short_title_value = entry_shorttitel.get()
        description_value = entry_description.get()
        insert_query = "INSERT INTO Location (Shorttitle, Description) VALUES (?, ?)"
        execute_query(insert_query, (short_title_value, description_value))
        load_locations()
        entry_id.config(state='normal')
        entry_id.delete(0, 'end')
        entry_id.config(state='readonly')
        entry_shorttitel.delete(0, 'end')
        entry_description.delete(0, 'end')

    def delete_location():
        selected_location = option_var.get()
        if selected_location:
            location_id = int(selected_location.split()[0])
            delete_query = "DELETE FROM Location WHERE ID = ?"
            execute_query(delete_query, (location_id,))
            load_locations()
            entry_id.config(state='normal')
            entry_id.delete(0, 'end')
            entry_id.config(state='readonly')
            entry_shorttitel.delete(0, 'end')
            entry_description.delete(0, 'end')

    def load_locations():
        query = "SELECT ID, Shorttitle, Description FROM Location"
        locations = execute_query_select(query)
        location_options = [f"{location[0]}" for location in locations]
        option_var.set("")
        option_menu['menu'].delete(0, 'end')
        for option in location_options:
            command = lambda value=option: option_var.set(value)
            option_menu['menu'].add_command(label=option, command=command)

    def select_location():
        selected_location = option_var.get()
        if selected_location:
            location_id = int(selected_location.split()[0])
            query = "SELECT ID, Shorttitle, Description FROM Location WHERE ID=?"
            selected_location = execute_query_select(query, (location_id,))
            if selected_location:
                entry_id.config(state='normal')
                entry_id.delete(0, 'end')
                entry_id.insert(0, str(selected_location[0][0]))
                entry_id.config(state='readonly')
                entry_shorttitel.delete(0, 'end')
                entry_shorttitel.insert(0, selected_location[0][1])
                entry_description.delete(0, 'end')
                entry_description.insert(0, selected_location[0][2])

    new_window = tk.Toplevel(root)
    new_window.title("Create Location")

    label_select = tk.Label(new_window, text="Select Location")
    label_select.grid(row=0, column=0, pady=5, padx=5, sticky="e")
    option_var = tk.StringVar(new_window)
    option_menu = tk.OptionMenu(new_window, option_var, "")
    option_menu.grid(row=0, column=1, pady=5, padx=5, sticky="w")

    label_shorttitel = tk.Label(new_window, text="Shorttitle")
    label_shorttitel.grid(row=0, column=2, pady=5, padx=5, sticky="e")
    entry_shorttitel = tk.Entry(new_window)
    entry_shorttitel.grid(row=0, column=3, pady=5, padx=5, sticky="w")

    label_description = tk.Label(new_window, text="Description")
    label_description.grid(row=0, column=4, pady=5, padx=5, sticky="e")
    entry_description = tk.Entry(new_window)
    entry_description.grid(row=0, column=5, pady=5, padx=5, sticky="w")

    label_id = tk.Label(new_window, text="ID")
    label_id.grid(row=0, column=6, pady=5, padx=5, sticky="e")
    entry_id = tk.Entry(new_window, state="readonly")
    entry_id.grid(row=0, column=7, pady=5, padx=5, sticky="w")

    btn_create_location = tk.Button(new_window, text="Create Location", command=create_location)
    btn_create_location.grid(row=1, column=0, columnspan=3, pady=10, padx=5, sticky="nsew")

    btn_select_location = tk.Button(new_window, text="Load Location", command=select_location)
    btn_select_location.grid(row=1, column=3, columnspan=3, pady=10, padx=5, sticky="nsew")

    btn_delete_location = tk.Button(new_window, text="Delete Location", command=delete_location)
    btn_delete_location.grid(row=1, column=6, columnspan=3, pady=10, padx=5, sticky="nsew")

    load_locations()


def enter_observation_window():

    def validate_time(time):
        try:
            formatted_time = datetime.strptime(time, "%H:%M").time()
            return formatted_time
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid time in HH:MM format")

    def load_locations(option_var, option_menu):
        query = "SELECT ID, Shorttitle FROM Location"
        locations = execute_query_select(query)
        location_options = [f"{location[0]} - {location[1]}" for location in locations]
        option_var.set("")
        option_menu['menu'].delete(0, 'end')
        for option in location_options:
            command = lambda value=option: option_var.set(value)
            option_menu['menu'].add_command(label=option, command=command)

    def load_animals(option_var, option_menu):
        query = "SELECT ID, Genus FROM Animal"
        animals = execute_query_select(query)
        animal_options = [f"{animal[0]} - {animal[1]}" for animal in animals]
        option_var.set("")
        option_menu['menu'].delete(0, 'end')
        for option in animal_options:
            command = lambda value=option: option_var.set(value)
            option_menu['menu'].add_command(label=option, command=command)

    def save_observation():
        time = entry_time.get()
        valid_time = validate_time(time)
        if valid_time:
            animal_id = option_var_animal.get()
            animal_id = int(animal_id.split('-')[0])
            location_id = option_var_location.get()
            location_id = int(location_id.split('-')[0])
            date = cal.get_date()
            gender = entry_gender.get()
            age = entry_age.get()
            weight = entry_weight.get()
            size = entry_size.get()
            query = "INSERT INTO Observation (AnimalID, LocationID, Date, Time, Gender, Age, Weight, Size) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
            parameters = (animal_id, location_id, date, time, gender, age, weight, size)
            execute_query(query, parameters)
            clear_fields()
            load_animals(option_var_animal, option_menu_animal)
            load_locations(option_var_location, option_menu_location)

    def clear_fields():
        option_var_animal.set("")
        option_var_location.set("")
        cal.delete(0, "end")
        entry_time.delete(0, "end")
        entry_gender.delete(0, "end")
        entry_age.delete(0, "end")
        entry_weight.delete(0, "end")
        entry_size.delete(0, "end")

    def reload_values():
        load_animals(option_var_animal, option_menu_animal)
        load_locations(option_var_location, option_menu_location)

    def delete_observation():
        pass

    new_window = tk.Toplevel(root)
    new_window.title("Enter Observation")

    label_select_animal = tk.Label(new_window, text="Animal")
    label_select_animal.grid(row=0, column=0, pady=5, padx=5, sticky="e")
    option_var_animal = tk.StringVar(new_window)
    option_menu_animal = tk.OptionMenu(new_window, option_var_animal, "")
    option_menu_animal.grid(row=0, column=1, pady=5, padx=5, sticky="w")

    label_select_location = tk.Label(new_window, text="Location")
    label_select_location.grid(row=1, column=0, pady=5, padx=5, sticky="e")
    option_var_location = tk.StringVar(new_window)
    option_menu_location = tk.OptionMenu(new_window, option_var_location, "")
    option_menu_location.grid(row=1, column=1, pady=5, padx=5, sticky="w")

    label_date = tk.Label(new_window, text="Date")
    label_date.grid(row=0, column=2, pady=5, padx=5, sticky="e")
    cal = DateEntry(new_window, width=12, background='gray', foreground='white', borderwidth=2)
    cal.grid(row=0, column=3, pady=5, padx=5, sticky="w")

    label_time = tk.Label(new_window, text="Time")
    label_time.grid(row=1, column=2, pady=5, padx=5, sticky="e")
    entry_time = tk.Entry(new_window)
    entry_time.grid(row=1, column=3, pady=5, padx=5, sticky="w")

    label_gender = tk.Label(new_window, text="Gender")
    label_gender.grid(row=0, column=4, pady=5, padx=5, sticky="e")
    entry_gender = tk.Entry(new_window)
    entry_gender.grid(row=0, column=5, pady=5, padx=5, sticky="w")

    label_age = tk.Label(new_window, text="Estimated Age")
    label_age.grid(row=1, column=4, pady=5, padx=5, sticky="e")
    entry_age = tk.Entry(new_window)
    entry_age.grid(row=1, column=5, pady=5, padx=5, sticky="w")

    label_weight = tk.Label(new_window, text="Estimated Weight")
    label_weight.grid(row=0, column=6, pady=5, padx=5, sticky="e")
    entry_weight = tk.Entry(new_window)
    entry_weight.grid(row=0, column=7, pady=5, padx=5, sticky="w")

    label_size = tk.Label(new_window, text="Estimated Size")
    label_size.grid(row=1, column=6, pady=5, padx=5, sticky="e")
    entry_size = tk.Entry(new_window)
    entry_size.grid(row=1, column=7, pady=5, padx=5, sticky="w")

    btn_save_observation = tk.Button(new_window, text="Reload", command=reload_values)
    btn_save_observation.grid(row=2, column=0, columnspan=2, pady=10, padx=5, sticky="nsew")

    btn_save_observation = tk.Button(new_window, text="Save Observation", command=save_observation)
    btn_save_observation.grid(row=2, column=2, columnspan=2, pady=10, padx=5, sticky="nsew")

    btn_clear_observation = tk.Button(new_window, text="Clear Input Fields", command=clear_fields)
    btn_clear_observation.grid(row=2, column=4, columnspan=2, pady=10, padx=5, sticky="nsew")

    btn_delete_observation = tk.Button(new_window, text="Delete Observation", command=delete_observation)
    btn_delete_observation.grid(row=2, column=6, columnspan=2, pady=10, padx=5, sticky="nsew")

    load_animals(option_var_animal, option_menu_animal)
    load_locations(option_var_location, option_menu_location)


def exit_application():
    root.destroy()


if __name__ == "__main__":
    main()
