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

int Task::_id = 0; // Initialize static variable

int main()
{

    vector<string> week = {"Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"};
    vector<string> months = {"Jan", "Feb", "Mar", "Apr", "May", "June", "July", "Aug", "Sept", "Oct", "Nov", "Dec"};

    Month *monthTasks[12];
    Week *weekTasks[7];


    for (int i = 0; i < 12; i++)
        monthTasks[i] = new Month("", 0, months[i]);

    for (int i = 0; i < 7; i++)
        weekTasks[i] = new Week("", 0, week[i]);

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
                cout << i + 1 << ". " << monthTasks[i]->month << " - Priority: " << monthTasks[i]->priority
                     << " - " << monthTasks[i]->description << endl;

            cout << "\n--- WEEKLY TASKS ---\n";
            for (int i = 0; i < 7; i++)
                cout << i + 1 << ". " << weekTasks[i]->week << " - Priority: " << weekTasks[i]->priority
                     << " - " << weekTasks[i]->description << endl;
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

            editPriority(*monthTasks[idx - 1], p);
            editDescription(*monthTasks[idx - 1], desc);
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

            editPriority(*weekTasks[idx - 1], p);
            editDescription(*weekTasks[idx - 1], desc);
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
    for (int i = 0; i < 12; i++)
        delete monthTasks[i];
    for (int i = 0; i < 7; i++)
        delete weekTasks[i];

    return 0;
}