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

	GDCLASS(OnnxSession, Object);

	Ort::Session *m_session;

protected:
	static void _bind_methods();
	Vector<PackedFloat32Array> _run_internal(Vector<PackedFloat32Array> &inputs);
	// only construct via OnnxRunner
	~OnnxSession();

public:
	OnnxSession(Ort::Session *session);
	// should never get called
	OnnxSession(){};

	Variant OnnxSession::run(Variant input);

	uint32_t num_inputs();
	Vector<int64_t> input_shape(uint32_t idx);
	String input_name(uint32_t idx);

	uint32_t num_outputs();
	Vector<int64_t> output_shape(uint32_t idx);
	String output_name(uint32_t idx);
};
