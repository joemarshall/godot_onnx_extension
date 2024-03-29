#ifdef __MINGW64__
#define _stdcall __cdecl
#define _In_
#define _In_z_
#define _In_opt_
#define _In_opt_z_
#define _Out_
#define _Outptr_
#define _Out_opt_
#define _Inout_
#define _Inout_opt_
#define _Frees_ptr_opt_
#define _Ret_maybenull_
#define _Ret_notnull_
#define _Check_return_
#define _Outptr_result_maybenull_
#define _In_reads_(X)
#define _Inout_updates_(X)
#define _Out_writes_(X)
#define _Inout_updates_all_(X)
#define _Out_writes_bytes_all_(X)
#define _Out_writes_all_(X)
#define _Success_(X)
#define _Outptr_result_buffer_maybenull_(X)


#endif
#include "ort_exception_catcher.h"

#define ORT_NO_EXCEPTIONS
#define ORT_CXX_API_THROW(string,code) OrtExceptionCatcher::Report(string,code)
#include <onnxruntime_cxx_api.h>
