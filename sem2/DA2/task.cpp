#include "task.h"
#include <string>
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