#include "task.h"
#include <vector>
int main()
{

    vector<string> months = {"Jan", "Feb", "Mar", "Apr", "May", "June", "July", "Aug", "Sept", "Oct", "nov", "Dec"};
    // use cpp v11 else vector cant have aggrgate declaration itll throw error
    Month **MonthObjects = new Month *[12];
    for (int i = 0; i < 12; i++)
    {
        string currM = months[i]; // it doesnt waste space, will be auto optimised by compiler. wrote this for better clarity
        MonthObjects[i] = new Month(currM);
    }
}