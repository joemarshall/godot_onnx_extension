#include "ort_exception_catcher.h"
#include <thread>
static thread_local OrtExceptionCatcher *current_handler = NULL;

void OrtExceptionCatcher::Report(std::string str, int error)
{
    OrtExceptionCatcher::Report(str.c_str(),error);
}

void OrtExceptionCatcher::Report(char *str, int error)
{
    if (current_handler != NULL)
    {
        current_handler->errText = str;
        current_handler->errCode = error;
        current_handler->hasError = true;
    }
}

OrtExceptionCatcher::OrtExceptionCatcher()
{
    current_handler = this;
}

OrtExceptionCatcher::~OrtExceptionCatcher()
{
    current_handler = NULL;
}

bool OrtExceptionCatcher::HasError()
{
    return hasError;
}

String OrtExceptionCatcher::GetErrorString()
{
    return errText;
}

int OrtExceptionCatcher::GetErrorCode()
{
    return errCode;
}
