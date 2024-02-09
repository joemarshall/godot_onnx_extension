#include "ort_session.hpp"

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/utility_functions.hpp>
#include <godot_cpp/templates/vector.hpp>

#include <cassert>
#include <iostream>

#include "onnx_mingw_overrides.h"

using namespace godot;

class _ModelInfoInternal{
public:
	_ModelInfoInternal(Ort::Session* in_session)
		:session(in_session)
	{
		num_inputs = session->GetInputCount();
		for (size_t idx = 0; idx < num_inputs; idx++)
		{
			auto input_name = session->GetInputNameAllocated(idx, allocator);
			input_names.push_back(input_name.get());
			input_name_ptrs.push_back(std::move(input_name));

			auto type_info = session->GetInputTypeInfo(idx);
			auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
			auto input_node_dims = tensor_info.GetShape();
			input_shapes.push_back(input_node_dims);
			input_infos.push_back(tensor_info);
		}
		num_outputs = session->GetOutputCount();
		for (size_t idx = 0; idx < num_outputs; idx++)
		{
			auto output_name = session->GetOutputNameAllocated(idx, allocator);
			output_names.push_back(output_name.get());
			output_name_ptrs.push_back(std::move(output_name));

			auto type_info = session->GetOutputTypeInfo(idx);
			auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
			auto output_node_dims = tensor_info.GetShape();
			output_shapes.push_back(output_node_dims);
			output_infos.push_back(tensor_info);

		}


	}

	~_ModelInfoInternal()
	{
		delete session;
	}
	Ort::Session *session;
	Ort::AllocatorWithDefaultOptions allocator;
	std::vector<Ort::AllocatedStringPtr> input_name_ptrs;
	std::vector<Ort::AllocatedStringPtr> output_name_ptrs;
	std::vector<char*> input_names;
	std::vector<char*> output_names;
	std::vector<Ort::ConstTensorTypeAndShapeInfo> input_infos;
	std::vector<Ort::ConstTensorTypeAndShapeInfo> output_infos;
	int num_outputs;
	int num_inputs;
	std::vector<std::vector<int64_t>> input_shapes;
	std::vector<std::vector<int64_t>> output_shapes;
};

void OnnxSession::_bind_methods()
{

	ClassDB::bind_method(D_METHOD("run"), &OnnxSession::run);
	ClassDB::bind_method(D_METHOD("num_inputs"), &OnnxSession::num_inputs);
	ClassDB::bind_method(D_METHOD("input_shape"), &OnnxSession::input_shape);
	ClassDB::bind_method(D_METHOD("input_name"), &OnnxSession::input_name);
	ClassDB::bind_method(D_METHOD("num_outputs"), &OnnxSession::num_outputs);
	ClassDB::bind_method(D_METHOD("output_shape"), &OnnxSession::output_shape);
	ClassDB::bind_method(D_METHOD("output_name"), &OnnxSession::output_name);
}

OnnxSession::OnnxSession() : m_model(NULL), RefCounted()
{
}

void OnnxSession::connectOnnxSession(Ort::Session *session)
{
	m_model = new _ModelInfoInternal(session);
}

OnnxSession::~OnnxSession()
{
	if (m_model != NULL)
	{
		// session object should clean itself up on destruction
		delete m_model;
		m_model = NULL;
	}
}

uint32_t OnnxSession::num_inputs()
{
	return m_model->num_inputs;
}

PackedInt64Array OnnxSession::input_shape(uint32_t idx)
{
	ERR_FAIL_INDEX_V_MSG(idx, num_inputs(), PackedInt64Array(), vformat("Input index: %d is out of range (0--%d)", idx, num_inputs()));
	PackedInt64Array ret_dimensions;
	for (auto x : m_model->input_shapes[idx])
	{
		ret_dimensions.push_back(x);
	}
	return ret_dimensions;
}

String OnnxSession::input_name(uint32_t idx)
{
	ERR_FAIL_INDEX_V_MSG(idx, num_inputs(), String(), vformat("Input index: %d is out of range (0--%d)", idx, num_inputs()));
	String input_copy = String(m_model->input_names[idx]);
	return input_copy;
}

uint32_t OnnxSession::num_outputs()
{
	return m_model->num_outputs;
}

PackedInt64Array OnnxSession::output_shape(uint32_t idx)
{
	ERR_FAIL_INDEX_V_MSG(idx, num_outputs(), PackedInt64Array(), vformat("Output index: %d is out of range (0--%d)", idx, num_inputs()));
	PackedInt64Array ret_dimensions;
	for (auto x : m_model->output_shapes[idx])
	{
		ret_dimensions.push_back(x);
	}
	return ret_dimensions;
}

String OnnxSession::output_name(uint32_t idx)
{
	ERR_FAIL_INDEX_V_MSG(idx, num_outputs(), String(), vformat("Output index: %d is out of range (0--%d)", idx, num_inputs()));
	String output_copy = String(m_model->output_names[idx]);
	return output_copy;
}

Variant OnnxSession::run(Variant input)
{
	// input is either: PackedFloatArray (for single input), or Array<PackedFloatArray> for multi input
	Vector<PackedFloat32Array> inputs;
	auto type = input.get_type();
	if (type == Variant::ARRAY)
	{
		// multi-input
		// n.b. don't support plain arrays for single input
		Array arr = input;
		for (size_t i = 0; i < arr.size(); i++)
		{
			Variant this_arr = arr[i];
			// input should be PACKED_FLOAT32_ARRAY
			ERR_FAIL_COND_V_MSG(this_arr.get_type() != Variant::PACKED_FLOAT32_ARRAY, Variant(), "OnnxSession input data must be PackedFloat32Array or an Array of them");
			PackedFloat32Array as_arr = static_cast<PackedFloat32Array>(this_arr);
			inputs.push_back(as_arr);
		}
	}
	else if (type == Variant::PACKED_FLOAT32_ARRAY)
	{
		inputs.push_back((PackedFloat32Array)(input));
	}
	else
	{
		ERR_FAIL_V_MSG(Variant(), "Input needs to be either a single PackedFloat32Array, or an Array of them");
	}
	{
		OrtExceptionCatcher catcher;
		auto outputs = _run_internal(inputs);
		if(catcher.HasError())
		{
			ERR_FAIL_V_MSG(Variant(), vformat("Error in onnx runtime: %s (%d)",catcher.GetErrorString(),catcher.GetErrorCode()));
			return Variant();
		}
		Variant retval;
		if (outputs.size() == 1)
		{
			// if model has 1 output, return a packedfloat32array
			retval = Variant(outputs[0]);
		}
		else
		{
			// else return array of packedfloat32array
			// TODO: will this cast even work?
			auto a = Array();
			for (auto i : outputs)
			{
				Variant v = Variant(i);
				a.append(v);
			}
			retval = a;
		}
		return retval;
	}
	return Variant();
}

Vector<PackedFloat32Array> OnnxSession::_run_internal(Vector<PackedFloat32Array> &inputs)
{
	const size_t num_input_nodes = m_model->num_inputs;
	ERR_FAIL_COND_V_MSG(inputs.size() != num_input_nodes, Vector<PackedFloat32Array>(), "Wrong number of inputs for OnnxSession run");
	std::vector<Ort::Value> in_tensors;

	for (size_t idx = 0; idx < num_input_nodes; idx++)
	{
		auto tensor_info = m_model->input_infos[idx];
		auto input_node_dims = tensor_info.GetShape();
		auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
		auto element_count = tensor_info.GetElementCount();

		auto input = inputs[idx];
		ERR_FAIL_COND_V_MSG(input.size() != element_count, Vector<PackedFloat32Array>(), vformat("Input has incorrect size in OnnxSession.run %d, expected %d", input.size(), element_count));

		Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input.ptrw(), element_count,
																  input_node_dims.data(), input_node_dims.size());
		assert(input_tensor.IsTensor());
		in_tensors.push_back(std::move(input_tensor));
	}

	const size_t num_output_nodes = m_model->num_outputs;
	std::vector<const char *> output_node_names;
	// this second vector is used because otherwise output_name will be deallocated
	// at end of loop iteration below, whereas we want it deallocated after calling session.Run
	auto run_options = Ort::RunOptions();
	auto output_tensors =
		m_model->session->Run(run_options, m_model->input_names.data(), in_tensors.data(), in_tensors.size(), m_model->output_names.data(), m_model->output_names.size());
	assert(output_tensors.size() == num_output_nodes && output_tensors.front().IsTensor());

	Vector<PackedFloat32Array> out_vals;
	for (auto it = output_tensors.begin(); it != output_tensors.end(); ++it)
	{
		Ort::Value &out_tensor = *it;
		auto tensorShape = out_tensor.GetTensorTypeAndShapeInfo();
		size_t count = tensorShape.GetElementCount();
		float *floatarr = out_tensor.GetTensorMutableData<float>();
		PackedFloat32Array new_array;
		new_array.resize(count);
		memcpy(new_array.ptrw(), floatarr, sizeof(float) * count);
		out_vals.push_back(new_array);
	}
	return out_vals;
}
