#pragma once

#include <godot_cpp/classes/object.hpp>
#include <godot_cpp/core/class_db.hpp>

using namespace godot;

class OnnxSession;

class OnnxRunner : public Object
{
	GDCLASS(OnnxRunner, Object);

	static OnnxRunner *singleton;

	bool initialized_api;

protected:
	static void _bind_methods();
	void _init_api();

public:
	static OnnxRunner *get_singleton();

	OnnxSession* load_model(String model_source );

	OnnxRunner();
	~OnnxRunner();

	void hello_singleton();
};
