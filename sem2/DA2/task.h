#include <string>
using namespace std;

class Task
{
private:
    static int _id;

public:
    int priority; // from 0 to 2
    string description;
    string type; // "month" or "week"
    string name; // month name or weekday name

    inline void initStatic()
    {
        _id = 0;
    };
    Task() {}
    Task(int priority, string description) : priority(priority), description(description)
    {
        _id++;
    }
    Task(int priority, string description, string type, string name) : priority(priority), description(description), type(type), name(name)
    {
        _id++;
    }
};

class Month : public Task // monthly tasks
{
public:
    string month;
    friend void editDescription(Month &mon, string newDes);
    friend void editPriority(Month &mon, int priority);
    Month(string Month) : month(Month)
    {
        type = "month";
        name = Month;
    }
    Month(string description, int priority, string month) : Task(priority, description, "month", month), month(month) {}
};

class Week : public Task // weekly tasks
{
public:
    string week;
    friend void editDescription(Week &mon, string newDes);
    friend void editPriority(Week &mon, int priority);
    Week(string week) : week(week)
    {
        type = "week";
        name = week;
    }
    Week(string description, int priority, string week) : Task(priority, description, "week", week), week(week) {}
};
