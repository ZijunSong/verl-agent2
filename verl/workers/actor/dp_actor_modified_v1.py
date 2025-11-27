"""
这版 update_policy 改后会遇到服务器通信错误：
[36m(WorkerDict pid=3490362)[0m [rank1]:[E1118 20:32:23.955028747 ProcessGroupNCCL.cpp:629] [Rank 1] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=2635, OpType=ALLREDUCE, NumelIn=1, NumelOut=1, Timeout(ms)=1800000) ran for 1800024 milliseconds before timing out.
[36m(WorkerDict pid=3490362)[0m [rank1]:[E1118 20:32:23.957372583 ProcessGroupNCCL.cpp:2168] [PG ID 0 PG GUID 0(default_pg) Rank 1]  failure detected by watchdog at work sequence id: 2635 PG status: last enqueued work: 2635, last completed work: 2634
[36m(WorkerDict pid=3490362)[0m [rank1]:[E1118 20:32:23.957466207 ProcessGroupNCCL.cpp:667] Stack trace of the failed collective not found, potentially because FlightRecorder is disabled. You can enable it by setting TORCH_NCCL_TRACE_BUFFER_SIZE to a non-zero value.
[36m(WorkerDict pid=3490362)[0m [rank1]:[E1118 20:32:23.957487695 ProcessGroupNCCL.cpp:681] [Rank 1] Some NCCL operations have failed or timed out. Due to the asynchronous nature of CUDA kernels, subsequent GPU operations might run on corrupted/incomplete data.
[36m(WorkerDict pid=3490362)[0m [rank1]:[E1118 20:32:23.957506340 ProcessGroupNCCL.cpp:695] [Rank 1] To avoid data inconsistency, we are taking the entire process down.
[36m(WorkerDict pid=3490084)[0m [rank0]:[E1118 20:32:23.968558948 ProcessGroupNCCL.cpp:629] [Rank 0] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=2667, OpType=_REDUCE_SCATTER_BASE, NumelIn=46797824, NumelOut=23398912, Timeout(ms)=1800000) ran for 1800053 milliseconds before timing out.
[36m(WorkerDict pid=3490084)[0m [rank0]:[E1118 20:32:23.970332207 ProcessGroupNCCL.cpp:1895] [PG ID 0 PG GUID 0(default_pg) Rank 0] Process group watchdog thread terminated with exception: [Rank 0] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=2667, OpType=_REDUCE_SCATTER_BASE, NumelIn=46797824, NumelOut=23398912, Timeout(ms)=1800000) ran for 1800053 milliseconds before timing out.
[36m(WorkerDict pid=3490084)[0m Exception raised from checkTimeout at /pytorch/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:632 (most recent call first):
[36m(WorkerDict pid=3490084)[0m frame #0: c10::Error::Error(c10::SourceLocation, std::string) + 0x96 (0x76a732b6c1b6 in /home/zyc/miniconda3/envs/verl-agent3/lib/python3.12/site-packages/torch/lib/libc10.so)
[36m(WorkerDict pid=3490084)[0m frame #1: c10d::ProcessGroupNCCL::WorkNCCL::checkTimeout(std::optional<std::chrono::duration<long, std::ratio<1l, 1000l> > >) + 0x2b4 (0x76a6e0bfec74 in /home/zyc/miniconda3/envs/verl-agent3/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so)
[36m(WorkerDict pid=3490084)[0m frame #2: c10d::ProcessGroupNCCL::watchdogHandler() + 0x890 (0x76a6e0c007d0 in /home/zyc/miniconda3/envs/verl-agent3/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so)
[36m(WorkerDict pid=3490084)[0m frame #3: c10d::ProcessGroupNCCL::ncclCommWatchdog() + 0x14d (0x76a6e0c016ed in /home/zyc/miniconda3/envs/verl-agent3/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so)
[36m(WorkerDict pid=3490084)[0m frame #4: <unknown function> + 0xdf0e6 (0x76d6ad0ea0e6 in /home/zyc/miniconda3/envs/verl-agent3/bin/../lib/libstdc++.so.6)
[36m(WorkerDict pid=3490084)[0m frame #5: <unknown function> + 0x94ac3 (0x76d6afc94ac3 in /lib/x86_64-linux-gnu/libc.so.6)
[36m(WorkerDict pid=3490084)[0m frame #6: <unknown function> + 0x1268c0 (0x76d6afd268c0 in /lib/x86_64-linux-gnu/libc.so.6)
[36m(WorkerDict pid=3490084)[0m 
[36m(WorkerDict pid=3490084)[0m [2025-11-18 20:32:23,672 E 3490084 3490489] logging.cc:118: Unhandled exception: N3c1016DistBackendErrorE. what(): [PG ID 0 PG GUID 0(default_pg) Rank 0] Process group watchdog thread terminated with exception: [Rank 0] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=2667, OpType=_REDUCE_SCATTER_BASE, NumelIn=46797824, NumelOut=23398912, Timeout(ms)=1800000) ran for 1800053 milliseconds before timing out.
[36m(WorkerDict pid=3490084)[0m Exception raised from checkTimeout at /pytorch/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:632 (most recent call first):
[36m(WorkerDict pid=3490084)[0m frame #0: c10::Error::Error(c10::SourceLocation, std::string) + 0x96 (0x76a732b6c1b6 in /home/zyc/miniconda3/envs/verl-agent3/lib/python3.12/site-packages/torch/lib/libc10.so)
[36m(WorkerDict pid=3490084)[0m frame #1: c10d::ProcessGroupNCCL::WorkNCCL::checkTimeout(std::optional<std::chrono::duration<long, std::ratio<1l, 1000l> > >) + 0x2b4 (0x76a6e0bfec74 in /home/zyc/miniconda3/envs/verl-agent3/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so)
[36m(WorkerDict pid=3490084)[0m frame #2: c10d::ProcessGroupNCCL::watchdogHandler() + 0x890 (0x76a6e0c007d0 in /home/zyc/miniconda3/envs/verl-agent3/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so)
[36m(WorkerDict pid=3490084)[0m frame #3: c10d::ProcessGroupNCCL::ncclCommWatchdog() + 0x14d (0x76a6e0c016ed in /home/zyc/miniconda3/envs/verl-agent3/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so)
[36m(WorkerDict pid=3490084)[0m frame #4: <unknown function> + 0xdf0e6 (0x76d6ad0ea0e6 in /home/zyc/miniconda3/envs/verl-agent3/bin/../lib/libstdc++.so.6)
[36m(WorkerDict pid=3490084)[0m frame #5: <unknown function> + 0x94ac3 (0x76d6afc94ac3 in /lib/x86_64-linux-gnu/libc.so.6)
[36m(WorkerDict pid=3490084)[0m frame #6: <unknown function> + 0x1268c0 (0x76d6afd268c0 in /lib/x86_64-linux-gnu/libc.so.6)
[36m(WorkerDict pid=3490084)[0m 
[36m(WorkerDict pid=3490084)[0m Exception raised from ncclCommWatchdog at /pytorch/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:1901 (most recent call first):
[36m(WorkerDict pid=3490084)[0m frame #0: c10::Error::Error(c10::SourceLocation, std::string) + 0x96 (0x76a732b6c1b6 in /home/zyc/miniconda3/envs/verl-agent3/lib/python3.12/site-packages/torch/lib/libc10.so)
[36m(WorkerDict pid=3490084)[0m frame #1: <unknown function> + 0xe5c6fc (0x76a6e085c6fc in /home/zyc/miniconda3/envs/verl-agent3/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so)
[36m(WorkerDict pid=3490084)[0m frame #2: <unknown function> + 0xdf0e6 (0x76d6ad0ea0e6 in /home/zyc/miniconda3/envs/verl-agent3/bin/../lib/libstdc++.so.6)
[36m(WorkerDict pid=3490084)[0m frame #3: <unknown function> + 0x94ac3 (0x76d6afc94ac3 in /lib/x86_64-linux-gnu/libc.so.6)
[36m(WorkerDict pid=3490084)[0m frame #4: <unknown function> + 0x1268c0 (0x76d6afd268c0 in /lib/x86_64-linux-gnu/libc.so.6)
[36m(WorkerDict pid=3490084)[0m 
[36m(WorkerDict pid=3490362)[0m [rank1]:[E1118 20:32:23.963589671 ProcessGroupNCCL.cpp:1895] [PG ID 0 PG GUID 0(default_pg) Rank 1] Process group watchdog thread terminated with exception: [Rank 1] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=2635, OpType=ALLREDUCE, NumelIn=1, NumelOut=1, Timeout(ms)=1800000) ran for 1800024 milliseconds before timing out.
[36m(WorkerDict pid=3490362)[0m 
[36m(WorkerDict pid=3490362)[0m [2025-11-18 20:32:23,665 E 3490362 3490490] logging.cc:118: Unhandled exception: N3c1016DistBackendErrorE. what(): [PG ID 0 PG GUID 0(default_pg) Rank 1] Process group watchdog thread terminated with exception: [Rank 1] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=2635, OpType=ALLREDUCE, NumelIn=1, NumelOut=1, Timeout(ms)=1800000) ran for 1800024 milliseconds before timing out.
[36m(WorkerDict pid=3490362)[0m 
[36m(WorkerDict pid=3490362)[0m 
[36m(WorkerDict pid=3490362)[0m [2025-11-18 20:32:23,675 E 3490362 3490490] logging.cc:125: Stack trace: 
[36m(WorkerDict pid=3490362)[0m  /home/zyc/miniconda3/envs/verl-agent3/lib/python3.12/site-packages/ray/_raylet.so(+0x152da9a) [0x7c3c5f72da9a] ray::operator<<()
[36m(WorkerDict pid=3490362)[0m /home/zyc/miniconda3/envs/verl-agent3/lib/python3.12/site-packages/ray/_raylet.so(+0x15309a2) [0x7c3c5f7309a2] ray::TerminateHandler()
[36m(WorkerDict pid=3490362)[0m /home/zyc/miniconda3/envs/verl-agent3/bin/../lib/libstdc++.so.6(+0xc2519) [0x7c3c5e0cd519] __cxxabiv1::__terminate()
[36m(WorkerDict pid=3490362)[0m /home/zyc/miniconda3/envs/verl-agent3/bin/../lib/libstdc++.so.6(_ZSt10unexpectedv+0) [0x7c3c5e0c7063] std::unexpected()
[36m(WorkerDict pid=3490362)[0m /home/zyc/miniconda3/envs/verl-agent3/bin/../lib/libstdc++.so.6(+0xc2512) [0x7c3c5e0cd512] __cxxabiv1::__terminate()
[36m(WorkerDict pid=3490362)[0m /home/zyc/miniconda3/envs/verl-agent3/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so(+0xe5c7aa) [0x7c0c90e5c7aa] c10d::ProcessGroupNCCL::ncclCommWatchdog()
[36m(WorkerDict pid=3490362)[0m /home/zyc/miniconda3/envs/verl-agent3/bin/../lib/libstdc++.so.6(+0xdf0e6) [0x7c3c5e0ea0e6] execute_native_thread_routine
[36m(WorkerDict pid=3490362)[0m /lib/x86_64-linux-gnu/libc.so.6(+0x94ac3) [0x7c3c60a94ac3]
[36m(WorkerDict pid=3490362)[0m /lib/x86_64-linux-gnu/libc.so.6(+0x1268c0) [0x7c3c60b268c0]
[36m(WorkerDict pid=3490362)[0m 
[36m(WorkerDict pid=3490362)[0m *** SIGABRT received at time=1763469143 on cpu 138 ***
[36m(WorkerDict pid=3490362)[0m PC: @     0x7c3c60a969fc  (unknown)  pthread_kill
[36m(WorkerDict pid=3490362)[0m     @     0x7c3c60a42520  (unknown)  (unknown)
[36m(WorkerDict pid=3490362)[0m [2025-11-18 20:32:23,675 E 3490362 3490490] logging.cc:474: *** SIGABRT received at time=1763469143 on cpu 138 ***
[36m(WorkerDict pid=3490362)[0m [2025-11-18 20:32:23,675 E 3490362 3490490] logging.cc:474: PC: @     0x7c3c60a969fc  (unknown)  pthread_kill
[36m(WorkerDict pid=3490362)[0m [2025-11-18 20:32:23,675 E 3490362 3490490] logging.cc:474:     @     0x7c3c60a42520  (unknown)  (unknown)
[36m(WorkerDict pid=3490362)[0m Fatal Python error: Aborted
[36m(WorkerDict pid=3490362)[0m 
[36m(WorkerDict pid=3490362)[0m 
[36m(WorkerDict pid=3490362)[0m Extension modules: msgpack._cmsgpack, google._upb._message, psutil._psutil_linux, psutil._psutil_posix, yaml._yaml, charset_normalizer.md, requests.packages.charset_normalizer.md, requests.packages.chardet.md, uvloop.loop, ray._raylet, numpy._core._multiarray_umath, numpy.linalg._umath_linalg, pyarrow.lib, numpy.random._common, numpy.random.bit_generator, numpy.random._bounded_integers, numpy.random._mt19937, numpy.random.mtrand, numpy.random._philox, numpy.random._pcg64, numpy.random._sfc64, numpy.random._generator, pandas._libs.tslibs.ccalendar, pandas._libs.tslibs.np_datetime, pandas._libs.tslibs.dtypes, pandas._libs.tslibs.base, pandas._libs.tslibs.nattype, pandas._libs.tslibs.timezones, pandas._libs.tslibs.fields, pandas._libs.tslibs.timedeltas, pandas._libs.tslibs.tzconversion, pandas._libs.tslibs.timestamps, pandas._libs.properties, pandas._libs.tslibs.offsets, pandas._libs.tslibs.strptime, pandas._libs.tslibs.parsing, pandas._libs.tslibs.conversion, pandas._libs.tslibs.period, pandas._libs.tslibs.vectorized, pandas._libs.ops_dispatch, pandas._libs.missing, pandas._libs.hashtable, pandas._libs.algos, pandas._libs.interval, pandas._libs.lib, pyarrow._compute, pandas._libs.ops, pandas._libs.hashing, pandas._libs.arrays, pandas._libs.tslib, pandas._libs.sparse, pandas._libs.internals, pandas._libs.indexing, pandas._libs.index, pandas._libs.writers, pandas._libs.join, pandas._libs.window.aggregations, pandas._libs.window.indexers, pandas._libs.reshape, pandas._libs.groupby, pandas._libs.json, pandas._libs.parsers, pandas._libs.testing, torch._C, torch._C._dynamo.autograd_compiler, torch._C._dynamo.eval_frame, torch._C._dynamo.guards, torch._C._dynamo.utils, torch._C._fft, torch._C._linalg, torch._C._nested, torch._C._nn, torch._C._sparse, torch._C._special, regex._regex, markupsafe._speedups, PIL._imaging, scipy._lib._ccallback_c, scipy.linalg._fblas, scipy.linalg._flapack, _cyutility, scipy._cyutility, scipy.linalg.cython_lapack, scipy.linalg._cythonized_array_utils, scipy.linalg._solve_toeplitz, scipy.linalg._decomp_lu_cython, scipy.linalg._matfuncs_schur_sqrtm, scipy.linalg._matfuncs_expm, scipy.linalg._linalg_pythran, scipy.linalg.cython_blas, scipy.linalg._decomp_update, scipy.sparse._sparsetools, _csparsetools, scipy.sparse._csparsetools, scipy.sparse.linalg._dsolve._superlu, scipy.sparse.linalg._eigen.arpack._arpack, scipy.sparse.linalg._propack._spropack, scipy.sparse.linalg._propack._dpropack, scipy.sparse.linalg._propack._cpropack, scipy.sparse.linalg._propack._zpropack, scipy.optimize._group_columns, scipy._lib.messagestream, scipy.optimize._trlib._trlib, scipy.optimize._lbfgsb, _moduleTNC, scipy.optimize._moduleTNC, scipy.optimize._slsqplib, scipy.optimize._minpack, scipy.optimize._lsq.givens_elimination, scipy.optimize._zeros, scipy._lib._uarray._uarray, scipy.special._ufuncs_cxx, scipy.special._ellip_harm_2, scipy.special._special_ufuncs, scipy.special._gufuncs, scipy.special._ufuncs, scipy.special._specfun, scipy.special._comb, scipy.linalg._decomp_interpolative, scipy.optimize._bglu_dense, scipy.optimize._lsap, scipy.spatial._ckdtree, scipy.spatial._qhull, scipy.spatial._voronoi, scipy.spatial._hausdorff, scipy.spatial._distance_wrap, scipy.spatial.transform._rotation, scipy.spatial.transform._rigid_transform, scipy.optimize._direct, PIL._imagingft, av._core, av.logging, av.bytesource, av.buffer, av.audio.format, av.error, av.dictionary, av.container.pyio, av.option, av.descriptor, av.format, av.utils, av.stream, av.container.streams, av.sidedata.motionvectors, av.sidedata.sidedata, av.opaque, av.packet, av.container.input, av.container.output, av.container.core, av.codec.context, av.video.format, av.video.reformatter, av.plane, av.video.plane, av.video.frame, av.video.stream, av.codec.hwaccel, av.codec.codec, av.frame, av.audio.layout, av.audio.plane, av.audio.frame, av.audio.stream, av.filter.link, av.filter.context, av.filter.graph, av.filter.filter, av.filter.loudnorm, av.audio.resampler, av.audio.codeccontext, av.audio.fifo, av.bitstream, av.video.codeccontext, pyarrow._json, zmq.backend.cython._zmq, msgspec._core
[36m(WorkerDict pid=3490084)[0m /home/zyc/miniconda3/envs/verl-agent3/bin/../lib/libstdc++.so.6(+0xc2519) [0x76d6ad0cd519] __cxxabiv1::__terminate()
[36m(WorkerDict pid=3490084)[0m /home/zyc/miniconda3/envs/verl-agent3/bin/../lib/libstdc++.so.6(+0xc2512) [0x76d6ad0cd512] __cxxabiv1::__terminate()
[36m(WorkerDict pid=3490084)[0m /home/zyc/miniconda3/envs/verl-agent3/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so(+0xe5c7aa) [0x76a6e085c7aa] c10d::ProcessGroupNCCL::ncclCommWatchdog()
[36m(WorkerDict pid=3490084)[0m /lib/x86_64-linux-gnu/libc.so.6(+0x94ac3) [0x76d6afc94ac3]
[36m(WorkerDict pid=3490084)[0m /lib/x86_64-linux-gnu/libc.so.6(+0x1268c0) [0x76d6afd268c0]
[36m(WorkerDict pid=3490084)[0m 
[36m(WorkerDict pid=3490084)[0m 
[36m(WorkerDict pid=3490084)[0m 
[36m(WorkerDict pid=3490084)[0m Extension modules: msgpack._cmsgpack, google._upb._message, psutil._psutil_linux, psutil._psutil_posix, yaml._yaml, charset_normalizer.md, requests.packages.charset_normalizer.md, requests.packages.chardet.md, uvloop.loop, ray._raylet, numpy._core._multiarray_umath, numpy.linalg._umath_linalg, pyarrow.lib, numpy.random._common, numpy.random.bit_generator, numpy.random._bounded_integers, numpy.random._mt19937, numpy.random.mtrand, numpy.random._philox, numpy.random._pcg64, numpy.random._sfc64, numpy.random._generator, pandas._libs.tslibs.ccalendar, pandas._libs.tslibs.np_datetime, pandas._libs.tslibs.dtypes, pandas._libs.tslibs.base, pandas._libs.tslibs.nattype, pandas._libs.tslibs.timezones, pandas._libs.tslibs.fields, pandas._libs.tslibs.timedeltas, pandas._libs.tslibs.tzconversion, pandas._libs.tslibs.timestamps, pandas._libs.properties, pandas._libs.tslibs.offsets, pandas._libs.tslibs.strptime, pandas._libs.tslibs.parsing, pandas._libs.tslibs.conversion, pandas._libs.tslibs.period, pandas._libs.tslibs.vectorized, pandas._libs.ops_dispatch, pandas._libs.missing, pandas._libs.hashtable, pandas._libs.algos, pandas._libs.interval, pandas._libs.lib, pyarrow._compute, pandas._libs.ops, pandas._libs.hashing, pandas._libs.arrays, pandas._libs.tslib, pandas._libs.sparse, pandas._libs.internals, pandas._libs.indexing, pandas._libs.index, pandas._libs.writers, pandas._libs.join, pandas._libs.window.aggregations, pandas._libs.window.indexers, pandas._libs.reshape, pandas._libs.groupby, pandas._libs.json, pandas._libs.parsers, pandas._libs.testing, torch._C, torch._C._dynamo.autograd_compiler, torch._C._dynamo.eval_frame, torch._C._dynamo.guards, torch._C._dynamo.utils, torch._C._fft, torch._C._linalg, torch._C._nested, torch._C._nn, torch._C._sparse, torch._C._special, regex._regex, markupsafe._speedups, PIL._imaging, scipy._lib._ccallback_c, scipy.linalg._fblas, scipy.linalg._flapack, _cyutility, scipy._cyutility, scipy.linalg.cython_lapack, scipy.linalg._cythonized_array_utils, scipy.linalg._solve_toeplitz, scipy.linalg._decomp_lu_cython, scipy.linalg._matfuncs_schur_sqrtm, scipy.linalg._matfuncs_expm, scipy.linalg._linalg_pythran, scipy.linalg.cython_blas, scipy.linalg._decomp_update, scipy.sparse._sparsetools, _csparsetools, scipy.sparse._csparsetools, scipy.sparse.linalg._dsolve._superlu, scipy.sparse.linalg._eigen.arpack._arpack, scipy.sparse.linalg._propack._spropack, scipy.sparse.linalg._propack._dpropack, scipy.sparse.linalg._propack._cpropack, scipy.sparse.linalg._propack._zpropack, scipy.optimize._group_columns, scipy._lib.messagestream, scipy.optimize._trlib._trlib, scipy.optimize._lbfgsb, _moduleTNC, scipy.optimize._moduleTNC, scipy.optimize._slsqplib, scipy.optimize._minpack, scipy.optimize._lsq.givens_elimination, scipy.optimize._zeros, scipy._lib._uarray._uarray, scipy.special._ufuncs_cxx, scipy.special._ellip_harm_2, scipy.special._special_ufuncs, scipy.special._gufuncs, scipy.special._ufuncs, scipy.special._specfun, scipy.special._comb, scipy.linalg._decomp_interpolative, scipy.optimize._bglu_dense, scipy.optimize._lsap, scipy.spatial._ckdtree, scipy.spatial._qhull, scipy.spatial._voronoi, scipy.spatial._hausdorff, scipy.spatial._distance_wrap, scipy.spatial.transform._rotation, scipy.spatial.transform._rigid_transform, scipy.optimize._direct, PIL._imagingft, av._core, av.logging, av.bytesource, av.buffer, av.audio.format, av.error, av.dictionary, av.container.pyio, av.option, av.descriptor, av.format, av.utils, av.stream, av.container.streams, av.sidedata.motionvectors, av.sidedata.sidedata, av.opaque, av.packet, av.container.input, av.container.output, av.container.core, av.codec.context, av.video.format, av.video.reformatter, av.plane, av.video.plane, av.video.frame, av.video.stream, av.codec.hwaccel, av.codec.codec, av.frame, av.audio.layout, av.audio.plane, av.audio.frame, av.audio.stream, av.filter.link, av.filter.context, av.filter.graph, av.filter.filter, av.filter.loudnorm, av.audio.resampler, av.audio.codeccontext, av.audio.fifo, av.bitstream, av.video.codeccontext, pyarrow._json, zmq.backend.cython._zmq, msgspec._core, multidict._multidict, yarl._quoting_c, propcache._helpers_c, aiohttp._http_writer, aiohttp._http_parser
[36m(WorkerDict pid=3490362)[0m , multidict._multidict, yarl._quoting_c, propcache._helpers_c, aiohttp._http_writer, aiohttp._http_parser, aiohttp._websocket.mask, aiohttp._websocket.reader_c, frozenlist._frozenlist, sentencepiece._sentencepiece, vllm.cumem_allocator, cuda_utils, __triton_launcher (total: 190)
[36m(WorkerDict pid=3490084)[0m , aiohttp._websocket.mask, aiohttp._websocket.reader_c, frozenlist._frozenlist, sentencepiece._sentencepiece, vllm.cumem_allocator, cuda_utils, __triton_launcher (total: 190)
[36m(WorkerDict pid=3490084)[0m /home/zyc/miniconda3/envs/verl-agent3/lib/python3.12/multiprocessing/resource_tracker.py:279: UserWarning: resource_tracker: There appear to be 1 leaked shared_memory objects to clean up at shutdown
[36m(WorkerDict pid=3490084)[0m   warnings.warn('resource_tracker: There appear to be %d '

[36m(WorkerDict pid=3490362)[0m Total steps: 1, num_warmup_steps: 0
[36m(WorkerDict pid=3490362)[0m Actor use_remove_padding=True
[36m(WorkerDict pid=3490362)[0m Actor use_fused_kernels=False
[36m(WorkerDict pid=3490084)[0m kwargs: {'n': 1, 'logprobs': 0, 'max_tokens': 64, 'detokenize': False, 'temperature': 1.0, 'top_k': -1, 'top_p': 1, 'ignore_eos': False}
[36m(WorkerDict pid=3490362)[0m WARNING 11-18 20:01:02 [arg_utils.py:1658] VLLM_ATTENTION_BACKEND=XFORMERS is not supported by the V1 Engine. Falling back to V0. We recommend to remove VLLM_ATTENTION_BACKEND=XFORMERS from your config in favor of the V1 Engine.
[36m(TaskRunner pid=3451596)[0m [INFO] - Starting training loop...
[36m(TaskRunner pid=3451596)[0m Using LocalLogger is deprecated. The constructor API will change 
[36m(TaskRunner pid=3451596)[0m Checkpoint tracker file does not exist: %s /home/zyc/songzijun/verl-agent/checkpoints/verl_agent_alfworld/grpo_qwen3_4b/latest_checkpointed_iteration.txt
[36m(TaskRunner pid=3451596)[0m Training from scratch
[36m(TaskRunner pid=3451596)[0m [INFO] - checkpoint loaded
[36m(TaskRunner pid=3451596)[0m [INFO] - Starting epochs...
[36m(TaskRunner pid=3451596)[0m [INFO] - 1. 构造 DataProto
[36m(TaskRunner pid=3451596)[0m [INFO] - 2. 拿出用于生成的部分 gen_batch
[36m(TaskRunner pid=3451596)[0m [INFO] - 3. 交给 traj_collector + env 做多轮交互，得到带 response 的 batch
[36m(TaskRunner pid=3451596)[0m [DEBUG] non_tensor_batch keys after multi_turn_loop: dict_keys(['anchor_obs', 'index', 'data_source', 'uid', 'traj_uid', 'raw_prompt', 'is_action_valid', 'rewards', 'active_masks', 'episode_rewards', 'episode_lengths', 'tool_callings', 'success_rate', 'pick_clean_then_place_in_recep_success_rate', 'look_at_obj_in_light_success_rate', 'pick_two_obj_and_place_success_rate', 'pick_and_place_success_rate', 'pick_heat_then_place_in_recep_success_rate', 'pick_cool_then_place_in_recep_success_rate'])
[36m(TaskRunner pid=3451596)[0m [DEBUG] uid: shape=(640,), first_5=['135e9cf8-d678-421e-b782-f4f025a5f36b'
[36m(TaskRunner pid=3451596)[0m  '135e9cf8-d678-421e-b782-f4f025a5f36b'
[36m(TaskRunner pid=3451596)[0m  '135e9cf8-d678-421e-b782-f4f025a5f36b'
[36m(TaskRunner pid=3451596)[0m  '135e9cf8-d678-421e-b782-f4f025a5f36b'
[36m(TaskRunner pid=3451596)[0m  '135e9cf8-d678-421e-b782-f4f025a5f36b']
[36m(TaskRunner pid=3451596)[0m [DEBUG] traj_uid: shape=(640,), first_5=['31fe0ae3-ddb6-429f-a9df-7fccd028db17'
[36m(TaskRunner pid=3451596)[0m  '31fe0ae3-ddb6-429f-a9df-7fccd028db17'
[36m(TaskRunner pid=3451596)[0m  '31fe0ae3-ddb6-429f-a9df-7fccd028db17'
[36m(TaskRunner pid=3451596)[0m  '31fe0ae3-ddb6-429f-a9df-7fccd028db17'
[36m(TaskRunner pid=3451596)[0m  '31fe0ae3-ddb6-429f-a9df-7fccd028db17']
[36m(TaskRunner pid=3451596)[0m [DEBUG] index: shape=(640,), first_5=[0 0 0 0 0]
[36m(TaskRunner pid=3451596)[0m [INFO] - 4. GiGPO 用到的 step 回报
[36m(TaskRunner pid=3451596)[0m [INFO] - 5. 一些 batch 的调整
[36m(TaskRunner pid=3451596)[0m [INFO] - 6. 计算 reward（rule-based / RM）
[36m(WorkerDict pid=3490362)[0m kwargs: {'n': 1, 'logprobs': 0, 'max_tokens': 64, 'detokenize': False, 'temperature': 1.0, 'top_k': -1, 'top_p': 1, 'ignore_eos': False}
[36m(TaskRunner pid=3451596)[0m [INFO] - 7. 重算old_log_prob，算熵
[36m(TaskRunner pid=3451596)[0m [INFO] - 8. ref policy log_prob（如果有）
[36m(TaskRunner pid=3451596)[0m [INFO] - 9. critic value（如果有）
[36m(TaskRunner pid=3451596)[0m list(reward_extra_infos_dict.keys())=[]
[36m(TaskRunner pid=3451596)[0m [INFO] - 10. 计算 advantage（GRPO / GAE / ...）
[36m(TaskRunner pid=3451596)[0m rewards.shape: (640, 64)
[36m(TaskRunner pid=3451596)[0m [DEBUG][fit] sft_lambda_cfg: 0.9
[36m(TaskRunner pid=3451596)[0m [INFO] - 11. update_critic（如果用 critic）
[36m(TaskRunner pid=3451596)[0m [DEBUG] - 当 RL 一个 group 里无法采到正样本的时候，将该 prompt 用 SFT 学习
[36m(TaskRunner pid=3451596)[0m [DEBUG] - 看看会跳转到哪里去更新 actor
[36m(WorkerDict pid=3490084)[0m [DEBUG][fsdp_workers] keys in data.batch: _StringKeys(dict_keys(['attention_mask', 'prompts', 'input_ids', 'position_ids', 'rollout_log_probs', 'responses', 'response_mask', 'old_log_probs', 'ref_log_prob', 'token_level_scores', 'token_level_rewards', 'advantages', 'returns', 'use_sft_loss']))
[36m(WorkerDict pid=3490084)[0m [DEBUG][fsdp_workers] use_sft_loss: 70 / 320
[36m(WorkerDict pid=3490084)[0m [DEBUG][dp_actor] sft_lambda: 0.9
[36m(WorkerDict pid=3490084)[0m [DEBUG][dp_actor] use_sft_loss found in data.batch
[33m(raylet)[0m A worker died or was killed while executing a task by an unexpected system error. To troubleshoot the problem, check the logs for the dead worker. RayTask ID: fffffffffffffffff1a990bd69473060e1fe19a201000000 Worker ID: 844534794edd787d475de9b886271da4c9266fb11375cc24ef84b174 Node ID: 0e9947ba391662aa35520c43ad3a07f6c0d7764f6cccfed4f972dcf0 Worker IP address: 172.18.132.17 Worker port: 34533 Worker PID: 3490362 Worker exit type: SYSTEM_ERROR Worker exit detail: Worker unexpectedly exits with a connection error code 2. End of file. There are some potential root causes. (1) The process is killed by SIGKILL by OOM killer due to high memory usage. (2) ray stop --force is called. (3) The worker is crashed unexpectedly due to SIGSEGV or other unexpected errors.
[36m(WorkerDict pid=3490362)[0m [DEBUG][fsdp_workers] keys in data.batch: _StringKeys(dict_keys(['attention_mask', 'prompts', 'input_ids', 'position_ids', 'rollout_log_probs', 'responses', 'response_mask', 'old_log_probs', 'ref_log_prob', 'token_level_scores', 'token_level_rewards', 'advantages', 'returns', 'use_sft_loss']))
[36m(WorkerDict pid=3490362)[0m [DEBUG][fsdp_workers] use_sft_loss: 100 / 320
[36m(WorkerDict pid=3490362)[0m [DEBUG][dp_actor] sft_lambda: 0.9
[36m(WorkerDict pid=3490362)[0m [DEBUG][dp_actor] use_sft_loss found in data.batch
Error executing job with overrides: ['algorithm.adv_estimator=grpo', 'data.train_files=/home/zyc/data/verl-agent/text/train.parquet', 'data.val_files=/home/zyc/data/verl-agent/text/test.parquet', 'data.train_batch_size=16', 'data.val_batch_size=128', 'data.max_prompt_length=2048', 'data.max_response_length=64', 'data.filter_overlong_prompts=True', 'data.truncation=left', 'data.return_raw_chat=True', 'actor_rollout_ref.model.path=/data/szj/models/Qwen/Qwen2.5-1.5B-Instruct', 'actor_rollout_ref.actor.optim.lr=1e-6', 'actor_rollout_ref.model.use_remove_padding=True', 'actor_rollout_ref.actor.ppo_mini_batch_size=16', 'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4', 'actor_rollout_ref.actor.use_kl_loss=True', 'actor_rollout_ref.actor.kl_loss_coef=0.01', 'actor_rollout_ref.actor.kl_loss_type=low_var_kl', 'actor_rollout_ref.model.enable_gradient_checkpointing=True', 'actor_rollout_ref.actor.fsdp_config.param_offload=False', 'actor_rollout_ref.actor.fsdp_config.optimizer_offload=False', 'actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32', 'actor_rollout_ref.rollout.tensor_model_parallel_size=2', 'actor_rollout_ref.rollout.name=vllm', 'actor_rollout_ref.rollout.gpu_memory_utilization=0.6', 'actor_rollout_ref.rollout.enable_chunked_prefill=False', 'actor_rollout_ref.rollout.enforce_eager=False', 'actor_rollout_ref.rollout.free_cache_engine=False', 'actor_rollout_ref.rollout.val_kwargs.temperature=0.4', 'actor_rollout_ref.rollout.val_kwargs.do_sample=True', 'actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32', 'actor_rollout_ref.ref.fsdp_config.param_offload=True', 'actor_rollout_ref.actor.use_invalid_action_penalty=True', 'actor_rollout_ref.actor.invalid_action_penalty_coef=0.1', 'algorithm.use_kl_in_reward=False', 'env.env_name=alfworld/AlfredTWEnv', 'env.seed=0', 'env.max_steps=5', 'env.rollout.n=8', 'env.resources_per_worker.num_cpus=0.1', 'trainer.critic_warmup=0', 'trainer.logger=[console,wandb]', 'trainer.project_name=verl_agent_alfworld', 'trainer.experiment_name=grpo_qwen3_4b', 'trainer.n_gpus_per_node=2', 'trainer.nnodes=1', 'trainer.save_freq=-1', 'trainer.test_freq=-1', 'trainer.total_epochs=1', 'trainer.val_before_train=False']
Traceback (most recent call last):
  File "/home/zyc/songzijun/verl-agent/verl/trainer/main_ppo.py", line 29, in main
    run_ppo(config)
  File "/home/zyc/songzijun/verl-agent/verl/trainer/main_ppo.py", line 41, in run_ppo
    ray.get(runner.run.remote(config))
  File "/home/zyc/miniconda3/envs/verl-agent3/lib/python3.12/site-packages/ray/_private/auto_init_hook.py", line 22, in auto_init_wrapper
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/home/zyc/miniconda3/envs/verl-agent3/lib/python3.12/site-packages/ray/_private/client_mode_hook.py", line 104, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/zyc/miniconda3/envs/verl-agent3/lib/python3.12/site-packages/ray/_private/worker.py", line 2882, in get
    values, debugger_breakpoint = worker.get_objects(object_refs, timeout=timeout)
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/zyc/miniconda3/envs/verl-agent3/lib/python3.12/site-packages/ray/_private/worker.py", line 968, in get_objects
    raise value.as_instanceof_cause()
ray.exceptions.RayTaskError(ActorDiedError): [36mray::TaskRunner.run()[39m (pid=3451596, ip=172.18.132.17, actor_id=e68467f21eedbfec08822b4501000000, repr=<main_ppo.TaskRunner object at 0x7238d8c7a9f0>)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/zyc/songzijun/verl-agent/verl/trainer/main_ppo.py", line 179, in run
    trainer.fit()
  File "/home/zyc/songzijun/verl-agent/verl/trainer/ppo/ray_trainer.py", line 1378, in fit
    actor_output = self.actor_rollout_wg.update_actor(batch)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/zyc/songzijun/verl-agent/verl/single_controller/ray/base.py", line 51, in __call__
    output = ray.get(output)
             ^^^^^^^^^^^^^^^
           ^^^^^^^^^^^^^^^^^^^
           ^^^^^^^^^^^^^^^^^^^^^
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ray.exceptions.ActorDiedError: The actor died unexpectedly before finishing this task.
	class_name: create_colocated_worker_cls.<locals>.WorkerDict
	actor_id: f1a990bd69473060e1fe19a201000000
	pid: 3490362
	name: Ax01RTWorkerDict_0:1
	namespace: 33f02d44-619d-41b3-8a5a-51da5e92158f
	ip: 172.18.132.17
The actor is dead because its worker process has died. Worker exit type: SYSTEM_ERROR Worker exit detail: Worker unexpectedly exits with a connection error code 2. End of file. There are some potential root causes. (1) The process is killed by SIGKILL by OOM killer due to high memory usage. (2) ray stop --force is called. (3) The worker is crashed unexpectedly due to SIGSEGV or other unexpected errors.

Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.
[36m(WorkerDict pid=3490084)[0m [rank0]:[E1118 20:32:23.969084696 ProcessGroupNCCL.cpp:2168] [PG ID 0 PG GUID 0(default_pg) Rank 0]  failure detected by watchdog at work sequence id: 2667 PG status: last enqueued work: 2689, last completed work: 2666
[36m(WorkerDict pid=3490084)[0m [rank0]:[E1118 20:32:23.969100690 ProcessGroupNCCL.cpp:667] Stack trace of the failed collective not found, potentially because FlightRecorder is disabled. You can enable it by setting TORCH_NCCL_TRACE_BUFFER_SIZE to a non-zero value.
[36m(WorkerDict pid=3490084)[0m [rank0]:[E1118 20:32:23.969104727 ProcessGroupNCCL.cpp:681] [Rank 0] Some NCCL operations have failed or timed out. Due to the asynchronous nature of CUDA kernels, subsequent GPU operations might run on corrupted/incomplete data.
[36m(WorkerDict pid=3490084)[0m [rank0]:[E1118 20:32:23.969108407 ProcessGroupNCCL.cpp:695] [Rank 0] To avoid data inconsistency, we are taking the entire process down.
[36m(WorkerDict pid=3490362)[0m Exception raised from checkTimeout at /pytorch/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:632 (most recent call first):[32m [repeated 2x across cluster][0m
[36m(WorkerDict pid=3490362)[0m frame #0: c10::Error::Error(c10::SourceLocation, std::string) + 0x96 (0x7c0ce826c1b6 in /home/zyc/miniconda3/envs/verl-agent3/lib/python3.12/site-packages/torch/lib/libc10.so)[32m [repeated 3x across cluster][0m
[36m(WorkerDict pid=3490362)[0m frame #1: c10d::ProcessGroupNCCL::WorkNCCL::checkTimeout(std::optional<std::chrono::duration<long, std::ratio<1l, 1000l> > >) + 0x2b4 (0x7c0c911fec74 in /home/zyc/miniconda3/envs/verl-agent3/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so)[32m [repeated 2x across cluster][0m
[36m(WorkerDict pid=3490362)[0m frame #3: c10d::ProcessGroupNCCL::ncclCommWatchdog() + 0x14d (0x7c0c912016ed in /home/zyc/miniconda3/envs/verl-agent3/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so)[32m [repeated 4x across cluster][0m
[36m(WorkerDict pid=3490362)[0m frame #4: <unknown function> + 0x1268c0 (0x7c3c60b268c0 in /lib/x86_64-linux-gnu/libc.so.6)[32m [repeated 10x across cluster][0m
[36m(WorkerDict pid=3490362)[0m Exception raised from ncclCommWatchdog at /pytorch/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:1901 (most recent call first):
[36m(WorkerDict pid=3490084)[0m [2025-11-18 20:32:23,682 E 3490084 3490489] logging.cc:125: Stack trace: 
[36m(WorkerDict pid=3490084)[0m  /home/zyc/miniconda3/envs/verl-agent3/lib/python3.12/site-packages/ray/_raylet.so(+0x152da9a) [0x76d6ae72da9a] ray::operator<<()
[36m(WorkerDict pid=3490084)[0m /home/zyc/miniconda3/envs/verl-agent3/lib/python3.12/site-packages/ray/_raylet.so(+0x15309a2) [0x76d6ae7309a2] ray::TerminateHandler()
[36m(WorkerDict pid=3490084)[0m /home/zyc/miniconda3/envs/verl-agent3/bin/../lib/libstdc++.so.6(_ZSt10unexpectedv+0) [0x76d6ad0c7063] std::unexpected()
[36m(WorkerDict pid=3490084)[0m /home/zyc/miniconda3/envs/verl-agent3/bin/../lib/libstdc++.so.6(+0xdf0e6) [0x76d6ad0ea0e6] execute_native_thread_routine
[36m(WorkerDict pid=3490084)[0m *** SIGABRT received at time=1763469143 on cpu 89 ***
[36m(WorkerDict pid=3490084)[0m PC: @     0x76d6afc969fc  (unknown)  pthread_kill
[36m(WorkerDict pid=3490084)[0m     @     0x76d6afc42520  (unknown)  (unknown)
[36m(WorkerDict pid=3490084)[0m [2025-11-18 20:32:23,682 E 3490084 3490489] logging.cc:474: *** SIGABRT received at time=1763469143 on cpu 89 ***
[36m(WorkerDict pid=3490084)[0m [2025-11-18 20:32:23,682 E 3490084 3490489] logging.cc:474: PC: @     0x76d6afc969fc  (unknown)  pthread_kill
[36m(WorkerDict pid=3490084)[0m [2025-11-18 20:32:23,682 E 3490084 3490489] logging.cc:474:     @     0x76d6afc42520  (unknown)  (unknown)
[36m(WorkerDict pid=3490084)[0m Fatal Python error: Aborted
"""

import itertools
import logging
import os
from typing import Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, compute_policy_loss_gspo, kl_penalty
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_name, get_torch_device, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad_and_slice_inputs, ulysses_pad
from verl.workers.actor import BasePPOActor

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        print(f"Actor use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = (
            torch.compile(verl_F.entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else verl_F.entropy_from_logits
        )
        self.device_name = get_device_name()

    def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature, calculate_entropy=calculate_entropy)
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        return log_probs, entropys

    def _compute_sft_loss(self, batch, temperature: float, use_sft: torch.Tensor):
        """
        最小可用版本：
        - 只对 use_sft=True 的样本做 SFT
        - 其它样本的 label 全部是 ignore_index (-100)
        """

        input_ids = batch["input_ids"]          # [B, T]
        attention_mask = batch["attention_mask"]
        position_ids = batch["position_ids"]
        responses = batch["responses"]          # [B, R]

        B, T = input_ids.size()
        R = responses.size(1)
        start = T - R   # 假设 response 在序列最后 R 个 token

        # 默认全 ignore
        labels = input_ids.new_full(input_ids.shape, -100)

        # 只对 use_sft=True 的样本，把 response 段设成 label
        use_sft = use_sft.to(input_ids.device)
        if use_sft.any():
            idx = use_sft.nonzero(as_tuple=False).squeeze(-1)  # [num_sft]
            labels[idx, start:] = responses[idx]

        outputs = self.actor_module(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        logits = outputs.logits / temperature

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        sft_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        return sft_loss


    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        multi_turn = data.meta_info.get("multi_turn", False)
        sft_lambda = float(data.meta_info.get("sft_lambda", 1.0))  # added by songzijun
        print(f"[DEBUG][dp_actor] sft_lambda: {sft_lambda}")

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
        if multi_turn:
            select_keys.append("loss_mask")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
            
        if "use_sft_loss" in data.batch: ### added by songzijun
            print("[DEBUG][dp_actor] use_sft_loss found in data.batch")
            select_keys.append("use_sft_loss")
            
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if has_multi_modal_inputs:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for data in micro_batches:
                    # Support all hardwares
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(get_torch_device().current_device()), **data.non_tensor_batch}
                    else:
                        data = data.to(get_torch_device().current_device())  # actor device is cpu when using offload
                    
                    # ====== 1) 准备 mask ======
                    use_sft = None
                    if "use_sft_loss" in data:
                        use_sft = data["use_sft_loss"].to(torch.bool)
                        print("[DEBUG][dp_actor] use_sft_loss:", use_sft.sum().item(), "/", use_sft.numel())

                    responses = data["responses"]
                    response_length = responses.size(1)
                    attention_mask = data["attention_mask"]
                    if multi_turn:
                        response_mask = data["loss_mask"][:, -response_length:]
                    else:
                        response_mask = attention_mask[:, -response_length:]

                    # RL 部分只用非 SFT 样本
                    if use_sft is not None:
                        rl_sample_mask = (~use_sft).float().unsqueeze(1)  # [B, 1]
                        response_mask_rl = response_mask * rl_sample_mask  # [B, R]，SFT 样本的 token mask 全 0
                    else:
                        response_mask_rl = response_mask
                        rl_sample_mask = None

                    # ====== 2) 先算 RL loss（对 RL 样本；如果一个都没有就让它为 0） ======
                    old_log_prob = data["old_log_probs"]
                    advantages = data["advantages"]

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    calculate_entropy = entropy_coeff != 0
                    entropy, log_prob = self._forward_micro_batch(
                        micro_batch=data,
                        temperature=temperature,
                        calculate_entropy=calculate_entropy,
                    )

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    if loss_mode == "vanilla":
                        policy_loss_fn = compute_policy_loss
                    elif loss_mode == "gspo":
                        policy_loss_fn = compute_policy_loss_gspo
                    else:
                        raise ValueError(f"Unsupported loss_mode: {loss_mode}")

                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask_rl,
                        cliprange=clip_ratio,
                        cliprange_low=clip_ratio_low,
                        cliprange_high=clip_ratio_high,
                        clip_ratio_c=clip_ratio_c,
                        loss_agg_mode=loss_agg_mode,
                    )

                    # 如果这一 micro-batch 里一个 RL 样本都没有，response_mask_rl 全 0，
                    # pg_loss 会非常接近 0，这里当它就是 RL_loss=0 即可。
                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask_rl, loss_agg_mode=loss_agg_mode)
                        rl_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        rl_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = data["ref_log_prob"]
                        kld = kl_penalty(
                            logprob=log_prob,
                            ref_logprob=ref_log_prob,
                            kl_penalty=self.config.kl_loss_type,
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask_rl, loss_agg_mode=loss_agg_mode)
                        rl_loss = rl_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    # ====== 3) 再算 SFT loss（只对 use_sft == True 的样本） ======
                    if use_sft is not None and use_sft.any():
                        sft_loss = self._compute_sft_loss(data, temperature, use_sft=use_sft)
                        metrics["actor/sft_loss"] = sft_loss.detach().item()
                    else:
                        sft_loss = torch.zeros_like(rl_loss)
        
                    # ====== 4) 用 λ 做加权，统一 backward ======
                    if self.config.use_dynamic_bsz:
                        scale = (len(data) / self.config.ppo_mini_batch_size)
                    else:
                        scale = 1.0 / self.gradient_accumulation

                    total_loss = (sft_lambda * sft_loss + (1.0 - sft_lambda) * rl_loss) * scale
                    total_loss.backward()

                    data_metrics = {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                        "actor/sft_lambda": sft_lambda,
                    }
                    append_to_dict(metrics, data_metrics)

                grad_norm = self._optimizer_step()
                data = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, data_metrics)
        self.actor_optimizer.zero_grad()
        return metrics
