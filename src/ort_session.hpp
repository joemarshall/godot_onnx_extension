#pragma once

#include <godot_cpp/classes/object.hpp>
#include <godot_cpp/core/class_db.hpp>

using namespace godot;

namespace Ort
{
	class Session;
};

// a single session connection (i.e. an instantiated model)
// which you can use to run inference with
class OnnxSession : public Object
{
	friend class OnnxRunner;

	GDCLASS(OnnxSession, Object);

	Ort::Session *m_session;

protected:
	static void _bind_methods();
	Vector<Vector<float_t>> _run_internal(Vector<Vector<float_t>> inputs);
	// only construct via OnnxRunner
	OnnxSession(Ort::Session &session);
	~OnnxSession();

public:
	// should never get called
	OnnxSession(){};

	int num_inputs();
	Vector<int> input_shape(int idx);
	String input_name(int idx);

	int num_outputs();
	Vector<int> output_shape(int idx);
	String output_name(int idx);
};
