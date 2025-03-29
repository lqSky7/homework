#include <string>
using namespace std;

class Task{
private:
    static inline int _id = 0;

public:
    int priority; // from 0 to 2
    string description;
    Task(){} // so if nothing give to task it doesnt throw error , we can add tasks later using friend functions
    Task(int priority, string description) : priority(priority), description(description){
        _id++;
    }
};

class Month: public Task //monthly tasks
{
    public:
    string month;
    friend void editDescription(string newDes);
    friend void editPriority(int priority);
    Month(string Month): month(Month){}
    Month(string description, int priority, string month): Task(priority, description), month(month){}

};

class Week: public Task // weekly tasks
{
    public:
    string week;
    friend void editDescription(string newDes);
    friend void editPriority(int priority);
    Week(string week): week(week){}
    Week(string description, int priority, string week): Task(priority, description), week(week){}
};
