import tkinter as tk


def create_animal_window():
    new_window = tk.Toplevel(root)
    new_window.title("Create Animal")
    tk.Label(new_window, text="ID").grid(row=1, column=0, pady=5, padx=5, sticky="e")
    tk.Entry(new_window, state="readonly").grid(row=1, column=1, pady=5, padx=5, sticky="w")
    tk.Label(new_window, text="Genus").grid(row=2, column=0, pady=5, padx=5, sticky="e")
    tk.Entry(new_window).grid(row=2, column=1, pady=5, padx=5, sticky="w")
    tk.Button(new_window, text="Create Animal").grid(row=3, column=0, columnspan=2, pady=10, sticky="nsew")
    tk.Label(new_window, text="ID").grid(row=1, column=2, pady=5, padx=5, sticky="e")
    tk.OptionMenu(new_window, tk.StringVar(new_window), "Option 1", "Option 2", "Option 3").grid(row=1, column=3, pady=5, padx=5, sticky="w")
    tk.Label(new_window, text="").grid(row=2, column=4)  # Invisible label
    tk.Button(new_window, text="Delete Animal").grid(row=3, column=2, columnspan=2, pady=10, sticky="nsew")


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
