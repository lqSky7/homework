#include "task.h"
#include <iostream>
#include <vector>

void editDescription(Month &mon, string newDes)
{
    mon.description = newDes;
}

void editPriority(Month &mon, int priority)
{
    mon.priority = priority;
}

void editDescription(Week &mon, string newDes)
{
    mon.description = newDes;
}

void editPriority(Week &mon, int priority)
{
    mon.priority = priority;
}

int Task::_id = 0; //static var init

int main()
{
    vector<string> week = {"Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"};
    vector<string> months = {"Jan", "Feb", "Mar", "Apr", "May", "June", "July", "Aug", "Sept", "Oct", "Nov", "Dec"};

    vector<Task *> tasks;

    for (int i = 0; i < 12; i++)
    {
        tasks.push_back(new Month("", 0, months[i]));
    }

    for (int i = 0; i < 7; i++)
    {
        tasks.push_back(new Week("", 0, week[i]));
    }

    int choice = 0;
    while (choice != 4)
    {
        cout << "\n1. View Tasks\n2. Edit Monthly Task\n3. Edit Weekly Task\n4. Exit\nChoice: ";
        cin >> choice;

        switch (choice)
        {
        case 1:
        {
            cout << "\n--- MONTHLY TASKS ---\n";
            for (int i = 0; i < 12; i++)
            {
                Task *t = tasks[i];
                cout << i + 1 << ". " << t->name << " - Priority: " << t->priority
                     << " - " << t->description << endl;
            }

            cout << "\n--- WEEKLY TASKS ---\n";
            for (int i = 12; i < 19; i++)
            {
                Task *t = tasks[i];
                cout << i - 11 << ". " << t->name << " - Priority: " << t->priority
                     << " - " << t->description << endl;
            }
            break;
        }
        case 2:
        {
            int idx, p;
            string desc;
            cout << "Month (1-12): ";
            cin >> idx;
            if (idx < 1 || idx > 12)
                break;

            cout << "New priority (0-2): ";
            cin >> p;
            cout << "New description: ";
            cin.ignore();
            getline(cin, desc);

            tasks[idx - 1]->priority = p;
            tasks[idx - 1]->description = desc;
            break;
        }
        case 3:
        {
            int idx, p;
            string desc;
            cout << "Day (1-7): ";
            cin >> idx;
            if (idx < 1 || idx > 7)
                break;

            cout << "New priority (0-2): ";
            cin >> p;
            cout << "New description: ";
            cin.ignore();
            getline(cin, desc);

            tasks[idx + 11]->priority = p;
            tasks[idx + 11]->description = desc;
            break;
        }
        case 4:
            cout << "Bye!\n";
            break;
        default:
            cout << "Invalid choice\n";
        }
    }

    // Cleanup
    for (auto task : tasks)
    {
        delete task;
    }

    return 0;
}