#pragma once

#include <godot_cpp/classes/object.hpp>
#include <godot_cpp/core/class_db.hpp>

using namespace godot;

class OnnxRunner : public Object
{
	GDCLASS(OnnxRunner, Object);

	static OnnxRunner *singleton;

	bool initialized_api;

protected:
	static void _bind_methods();

public:
	static OnnxRunner *get_singleton();

	OnnxRunner();
	~OnnxRunner();

	void hello_singleton();
};
