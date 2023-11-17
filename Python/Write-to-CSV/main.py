import csv
import datetime
from filelock import FileLock

data_file = 'data.csv'
audit_file = 'audit.csv'


def write_to_csv(data):
    with open(data_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(data)


def write_audit_trail(action, timestamp):
    with open(audit_file, 'a', newline='') as auditfile:
        writer = csv.writer(auditfile)
        writer.writerow([action, timestamp])


def add_data(data):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lock = FileLock(data_file + ".lock")
    with lock:
        write_to_csv(data)
        write_audit_trail('Added data', timestamp)


def get_all_data():
    with open(data_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        data = [row for row in reader]
    return data


def main():
    # Usage example: Add data and retrieve it
    data_to_add = ['Jane Doe', '25', 'jane@example.com']
    add_data(data_to_add)
    all_data = get_all_data()
    print("All Data:")
    for row in all_data:
        print(row)


if __name__ == "__main__":
    main()
