#include<thread>
#include<string>

#include <godot_cpp/classes/object.hpp>
#include <godot_cpp/templates/hash_map.hpp>

using namespace godot;


class OrtExceptionCatcher
{
public:
    OrtExceptionCatcher();
    ~OrtExceptionCatcher();

    bool HasError();
    String GetErrorString();
    int GetErrorCode();

    static void Report(std::string str, int error);
    static void Report(char *error, int code);

protected:
    String errText;
    int errCode;
    bool hasError;

};