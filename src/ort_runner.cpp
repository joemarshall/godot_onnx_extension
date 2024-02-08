#include "ort_runner.hpp"

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

#ifdef __MINGW64__
#define _stdcall __stdcall
#endif
#include <onnxruntime_cxx_api.h>

#include"ort_session.hpp"


#include<iostream>

using namespace godot;

OnnxRunner *OnnxRunner::singleton = nullptr;

void OnnxRunner::_bind_methods()
{
	ClassDB::bind_method(D_METHOD("load_model"), &OnnxRunner::load_model);
}

OnnxRunner *OnnxRunner::get_singleton()
{
	return singleton;
}

OnnxRunner::OnnxRunner()
{
	ERR_FAIL_COND(singleton != nullptr);
	singleton = this;
	initialized_api = false;
	std::cout << "INIT RUNNER" <<std::endl;
}

OnnxRunner::~OnnxRunner()
{
	ERR_FAIL_COND(singleton != this);
	delete env;
	singleton = nullptr;
}

void log_fn (void *param, OrtLoggingLevel severity, const char *category, const char *logid, const char *code_location, const char *message)
{
	std::cout <<message <<std::endl;
}

void OnnxRunner::_init_api()
{
	if(!initialized_api){
		#ifdef ORT_API_MANUAL_INIT
		Ort::InitApi();
		#endif
		env=new Ort::Env(ORT_LOGGING_LEVEL_VERBOSE, "test",log_fn,NULL);

		initialized_api=true;
	}
}



OnnxSession* OnnxRunner::load_model(String model_source )
{
	std::cout << "INIT API" <<std::endl;
	_init_api();
	std::cout << "DONE" <<std::endl;
	const auto& api = Ort::GetApi();
	OrtTensorRTProviderOptionsV2* tensorrt_options;

	Ort::SessionOptions session_options;
	session_options.SetIntraOpNumThreads(1);

	session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
	
	#ifdef _WIN32
	const wchar_t* model_path = model_source.wide_string().get_data();
	#else
	const char* model_path = model_source.ascii().get_data();
	#endif
	std::wcout << L"MODEL PATH:" << model_path <<std::endl;
	try
	{
		Ort::Session *session=new Ort::Session(*env, model_path, session_options);
		std::cout << "MADE SESSION!" <<std::endl;
		OnnxSession* newSession= memnew(OnnxSession);
		newSession->connectOnnxSession(session);
		return newSession;
	}
	catch(const std::exception& e)
	{
		std::cerr << e.what() << '\n';
		return NULL;
	}	
}

void OnnxRunner::hello_singleton()
{
	UtilityFunctions::print("Hello GDExtension Singleton!");
}
