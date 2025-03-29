#include "task.h"
#include <string>
#include <vector>

int main()
{
    vector<string> week = {"Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"};
    vector<string> months = {"Jan", "Feb", "Mar", "Apr", "May", "June", "July", "Aug", "Sept", "Oct", "nov", "Dec"};
    // use cpp v11 else vector cant have aggrgate declaration itll throw error
    Month **MonthObjects = new Month *[12];
    for (int i = 0; i < 12; i++)
    {
        string currM = months[i]; // it doesnt waste space, will be auto optimised by compiler. wrote this for better clarity
        MonthObjects[i] = new Month(currM);
    }

    Week **WeekObjects = new Week *[7];
    for (int i = 0; i < 7; i++)
    {
        string currW = week[i];
        WeekObjects[i] = new Week(currW);
    }
}