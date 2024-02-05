#include "ort_singleton.hpp"

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

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
	singleton = nullptr;
}

void OnnxRunner::_init_api()
{
	if(!initialized_api){
		#ifdef ORT_API_MANUAL_INIT
		Ort::InitApi();
		#endif
		initialized_api=true;
	}
}

OnnxSession* OnnxRunner::load_model(String model_source )
{
	_init_api();
	Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
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
	Ort::Session *session=new Ort::Session(env, model_path, session_options);
	return new OnnxSession(session);
}

void OnnxRunner::hello_singleton()
{
	UtilityFunctions::print("Hello GDExtension Singleton!");
}
