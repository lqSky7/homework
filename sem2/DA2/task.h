#include <string>
using namespace std;

class Task
{
private:
    static int _id;

public:
    int priority; // from 0 to 2
    string description;
    inline void initStatic()
    {
        _id = 0;
    };
    Task() {} // so if nothing give to task it doesnt throw error , we can add tasks later using friend functions
    Task(int priority, string description) : priority(priority), description(description)
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
    Month(string Month) : month(Month) {}
    Month(string description, int priority, string month) : Task(priority, description), month(month) {}
};

class Week : public Task // weekly tasks
{
public:
    string week;
    friend void editDescription(Week &mon, string newDes);
    friend void editPriority(Week &mon, int priority);
    Week(string week) : week(week) {}
    Week(string description, int priority, string week) : Task(priority, description), week(week) {}
};
