#pragma once

/*
	This file is part of cpp-ethereum.

	cpp-ethereum is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	cpp-ethereum is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with cpp-ethereum.  If not, see <http://www.gnu.org/licenses/>.
*/
/** @file MinerAux.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 * CLI module for mining.
 */

#include <thread>
#include <chrono>
#include <fstream>
#include <iostream>
#include <signal.h>
#include <random>

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/trim_all.hpp>
#include <boost/optional.hpp>

#include <libdevcore/FileSystem.h>
#include <libevmcore/Instruction.h>
#include <libdevcore/StructuredLogger.h>
#include <libethcore/Exceptions.h>
#include <libdevcore/SHA3.h>
#include <libdevcore/CommonJS.h>
#include <libethcore/EthashAux.h>
#include <libethcore/EthashCUDAMiner.h>
#include <libethcore/EthashGPUMiner.h>
#include <libethcore/EthashCPUMiner.h>
#include <libethcore/Farm.h>
#if ETH_ETHASHCL || !ETH_TRUE
#include <libethash-cl/ethash_cl_miner.h>
#endif
#if ETH_ETHASHCUDA || !ETH_TRUE
#include <libethash-cuda/ethash_cuda_miner.h>
#endif
#if ETH_JSONRPC || !ETH_TRUE
#include <libweb3jsonrpc/WebThreeStubServer.h>
#include <jsonrpccpp/server/connectors/httpserver.h>
#include <jsonrpccpp/client/connectors/httpclient.h>
#endif
#include "BuildInfo.h"
#if ETH_JSONRPC || !ETH_TRUE
#include "PhoneHome.h"
#include "FarmClient.h"
#endif
using namespace std;
using namespace dev;
using namespace dev::eth;
using namespace boost::algorithm;
using dev::eth::Instruction;

#undef RETURN

bool isTrue(std::string const& _m)
{
	return _m == "on" || _m == "yes" || _m == "true" || _m == "1";
}

bool isFalse(std::string const& _m)
{
	return _m == "off" || _m == "no" || _m == "false" || _m == "0";
}

inline std::string credits()
{
	std::ostringstream out;
	out
		<< "Ethereum (++) " << dev::Version << endl
		<< "  Code by Gav Wood et al, (c) 2013, 2014, 2015." << endl;
	return out.str();
}

class BadArgument: public Exception {};
struct MiningChannel: public LogChannel
{
	static const char* name() { return EthGreen "miner"; }
	static const int verbosity = 2;
	static const bool debug = false;
};
#define minelog clog(MiningChannel)

class MinerCLI
{
public:
	enum class OperationMode
	{
		None,
		DAGInit,
		Benchmark,
		Simulation,
		Farm
	};


	MinerCLI(OperationMode _mode = OperationMode::None): mode(_mode) {}

	bool interpretOption(int& i, int argc, char** argv)
	{
		string arg = argv[i];
		if ((arg == "-F" || arg == "--farm") && i + 1 < argc)
		{
			mode = OperationMode::Farm;
			m_farmURL = argv[++i];
		}
		else if (arg == "--farm-recheck" && i + 1 < argc)
			try {
				m_farmRecheckPeriod = stol(argv[++i]);
			}
			catch (...)
			{
				cerr << "Bad " << arg << " option: " << argv[i] << endl;
				BOOST_THROW_EXCEPTION(BadArgument());
			}
		else if (arg == "--opencl-platform" && i + 1 < argc)
			try {
				m_openclPlatform = stol(argv[++i]);
			}
			catch (...)
			{
				cerr << "Bad " << arg << " option: " << argv[i] << endl;
				BOOST_THROW_EXCEPTION(BadArgument());
			}
		else if (arg == "--opencl-device" && i + 1 < argc)
			try {
				m_openclDevice = stol(argv[++i]);
				m_miningThreads = 1;
			}
			catch (...)
			{
				cerr << "Bad " << arg << " option: " << argv[i] << endl;
				BOOST_THROW_EXCEPTION(BadArgument());
			}
#if ETH_ETHASHCL || ETH_ETHASHCUDA || !ETH_TRUE
		else if ((arg == "--cl-global-work" || arg == "--cuda-grid-size")  && i + 1 < argc)
			try {
				m_globalWorkSizeMultiplier = stol(argv[++i]);
			}
			catch (...)
			{
				cerr << "Bad " << arg << " option: " << argv[i] << endl;
				BOOST_THROW_EXCEPTION(BadArgument());
			}
		else if ((arg == "--cl-local-work" || arg == "--cuda-block-size") && i + 1 < argc)
			try {
				m_localWorkSize = stol(argv[++i]);
			}
			catch (...)
			{
				cerr << "Bad " << arg << " option: " << argv[i] << endl;
				BOOST_THROW_EXCEPTION(BadArgument());
			}
		else if (arg == "--list-devices")
			m_shouldListDevices = true;
		else if ((arg == "--cl-extragpu-mem" || arg == "--cuda-extragpu-mem") && i + 1 < argc)
			m_extraGPUMemory = 1000000 * stol(argv[++i]);
#endif
#if ETH_ETHASHCL || !ETH_TRUE
		else if (arg == "--cl-ms-per-batch" && i + 1 < argc)
			try 
			{
				m_msPerBatch = stol(argv[++i]);
			}
			catch (...)
			{
				cerr << "Bad " << arg << " option: " << argv[i] << endl;
				BOOST_THROW_EXCEPTION(BadArgument());
			}
		else if (arg == "--allow-opencl-cpu")
			m_clAllowCPU = true;
#endif
#if ETH_ETHASHCUDA || !ETH_TRUE
		else if (arg == "--cuda-devices") 
		{
			while (m_cudaDeviceCount < 16 && i + 1 < argc)
			{
				try 
				{
					m_cudaDevices[m_cudaDeviceCount++] = stol(argv[++i]);
				}
				catch (...)
				{
					break;
				}
			}
		}
		else if (arg == "--cuda-turbo") 
			m_cudaHighCPULoad = true;
		else if (arg == "--cuda-streams" && i + 1 < argc)
			m_numStreams = stol(argv[++i]);
#endif
		else if (arg == "--phone-home" && i + 1 < argc)
		{
			string m = argv[++i];
			if (isTrue(m))
				m_phoneHome = true;
			else if (isFalse(m))
				m_phoneHome = false;
			else
			{
				cerr << "Bad " << arg << " option: " << m << endl;
				BOOST_THROW_EXCEPTION(BadArgument());
			}
		}
		else if (arg == "--benchmark-warmup" && i + 1 < argc)
			try {
				m_benchmarkWarmup = stol(argv[++i]);
			}
			catch (...)
			{
				cerr << "Bad " << arg << " option: " << argv[i] << endl;
				BOOST_THROW_EXCEPTION(BadArgument());
			}
		else if (arg == "--benchmark-trial" && i + 1 < argc)
			try {
				m_benchmarkTrial = stol(argv[++i]);
			}
			catch (...)
			{
				cerr << "Bad " << arg << " option: " << argv[i] << endl;
				BOOST_THROW_EXCEPTION(BadArgument());
			}
		else if (arg == "--benchmark-trials" && i + 1 < argc)
			try 
			{
				m_benchmarkTrials = stol(argv[++i]);
			}
			catch (...)
			{
				cerr << "Bad " << arg << " option: " << argv[i] << endl;
				BOOST_THROW_EXCEPTION(BadArgument());
			}
		else if (arg == "-C" || arg == "--cpu")
			m_minerType = MinerType::CPU;
		else if (arg == "-G" || arg == "--opencl")
			m_minerType = MinerType::CL;
		else if (arg == "-U" || arg == "--cuda") 
		{
			m_minerType = MinerType::CUDA;

			cout << "Genoil's CUDA ethminer" << endl;
			cout << "=====================================================================" << endl;
			cout << "Forked from github.com/ethereum/cpp-ethereum" << endl;
			cout << "Ported from Tim Hughes' OpenCL kernel" << endl;
			cout << "With contributions from tpruvot and sp_ " << endl << endl;
			cout << "Please consider donating a tiny fraction of the extra performance to:" << endl;
			cout << "ETH: 0xb9310b185455f863f526dab3d245809f6854b4d" << endl;
			cout << "BTC: 1Nu2fMCEBjmnLzqb8qUJpKgq5RoEWFhNcW" << endl << endl;
		}
		else if (arg == "--current-block" && i + 1 < argc)
			m_currentBlock = stol(argv[++i]);
		else if (arg == "--no-precompute")
			m_precompute = false;
		else if (arg == "--precompute")
			m_precompute = true;
		else if ((arg == "-D" || arg == "--create-dag") && i + 1 < argc)
		{
			string m = boost::to_lower_copy(string(argv[++i]));
			mode = OperationMode::DAGInit;
			try
			{
				m_initDAG = stol(m);
			}
			catch (...)
			{
				cerr << "Bad " << arg << " option: " << m << endl;
				BOOST_THROW_EXCEPTION(BadArgument());
			}
		}
		else if ((arg == "-w" || arg == "--check-pow") && i + 4 < argc)
		{
			string m;
			try
			{
				Ethash::BlockHeader bi;
				m = boost::to_lower_copy(string(argv[++i]));
				h256 powHash(m);
				m = boost::to_lower_copy(string(argv[++i]));
				h256 seedHash;
				if (m.size() == 64 || m.size() == 66)
					seedHash = h256(m);
				else
					seedHash = EthashAux::seedHash(stol(m));
				m = boost::to_lower_copy(string(argv[++i]));
				bi.setDifficulty(u256(m));
				auto boundary = bi.boundary();
				m = boost::to_lower_copy(string(argv[++i]));
				bi.setNonce(h64(m));
				auto r = EthashAux::eval(seedHash, powHash, bi.nonce());
				bool valid = r.value < boundary;
				cout << (valid ? "VALID :-)" : "INVALID :-(") << endl;
				cout << r.value << (valid ? " < " : " >= ") << boundary << endl;
				cout << "  where " << boundary << " = 2^256 / " << bi.difficulty() << endl;
				cout << "  and " << r.value << " = ethash(" << powHash << ", " << bi.nonce() << ")" << endl;
				cout << "  with seed as " << seedHash << endl;
				if (valid)
					cout << "(mixHash = " << r.mixHash << ")" << endl;
				cout << "SHA3( light(seed) ) = " << sha3(EthashAux::light(bi.seedHash())->data()) << endl;
				exit(0);
			}
			catch (...)
			{
				cerr << "Bad " << arg << " option: " << m << endl;
				BOOST_THROW_EXCEPTION(BadArgument());
			}
		}
		else if (arg == "-M" || arg == "--benchmark")
			mode = OperationMode::Benchmark;
		else if (arg == "-S" || arg == "--simulation")
			mode = OperationMode::Simulation;
		else if ((arg == "-t" || arg == "--mining-threads") && i + 1 < argc)
		{
			try 
			{
				m_miningThreads = stol(argv[++i]);
			}
			catch (...)
			{
				cerr << "Bad " << arg << " option: " << argv[i] << endl;
				BOOST_THROW_EXCEPTION(BadArgument());
			}
		}
		else
			return false;
		return true;
	}

	void execute()
	{
		if (m_shouldListDevices)
		{
#if ETH_ETHASHCL || !ETH_TRUE
			if (m_minerType == MinerType::CL)
				EthashGPUMiner::listDevices();
#endif
#if ETH_ETHASHCUDA || !ETH_TRUE
			if (m_minerType == MinerType::CUDA)
				EthashCUDAMiner::listDevices();
#endif
			exit(0);
		}

		if (m_minerType == MinerType::CPU)
			EthashCPUMiner::setNumInstances(m_miningThreads);
		else if (m_minerType == MinerType::CL)
		{
#if ETH_ETHASHCL || !ETH_TRUE
			if (!EthashGPUMiner::configureGPU(
					m_localWorkSize,
					m_globalWorkSizeMultiplier,
					m_msPerBatch,
					m_openclPlatform,
					m_openclDevice,
					m_clAllowCPU,
					m_extraGPUMemory,
					m_currentBlock
				))
				exit(1);
			EthashGPUMiner::setNumInstances(m_miningThreads);
#else
			cerr << "Selected GPU mining without having compiled with -DETHASHCL=1" << endl;
			exit(1);
#endif
		}
		else if (m_minerType == MinerType::CUDA)
		{
#if ETH_ETHASHCUDA || !ETH_TRUE
			if (m_cudaDeviceCount == 0)
			{
				m_cudaDevices[0]  = 0;
				m_cudaDeviceCount = 1;
			}
			EthashCUDAMiner::setDevices(m_cudaDevices, m_cudaDeviceCount);
			m_miningThreads = m_cudaDeviceCount;
	
			EthashCUDAMiner::setNumInstances(m_miningThreads);
			if (!EthashCUDAMiner::configureGPU(
				m_localWorkSize,
				m_globalWorkSizeMultiplier,
				m_numStreams,
				m_extraGPUMemory,
				m_cudaHighCPULoad,
				m_currentBlock
				))
				exit(1);
#else
			cerr << "Selected CUDA mining without having compiled with -DETHASHCUDA=1 or -DBUNDLE=cudaminer" << endl;
			exit(1);
#endif
		}
		if (mode == OperationMode::DAGInit)
			doInitDAG(m_initDAG);
		else if (mode == OperationMode::Benchmark)
			doBenchmark(m_minerType, m_phoneHome, m_benchmarkWarmup, m_benchmarkTrial, m_benchmarkTrials);
		else if (mode == OperationMode::Farm)
			doFarm(m_minerType, m_farmURL, m_farmRecheckPeriod);
		else if (mode == OperationMode::Simulation)
			doSimulation(m_minerType);
	}

	static void streamHelp(ostream& _out)
	{
		_out
#if ETH_JSONRPC || !ETH_TRUE
			<< "Work farming mode:" << endl
			<< "    -F,--farm <url>  Put into mining farm mode with the work server at URL (default: http://127.0.0.1:8545)" << endl
			<< "    --farm-recheck <n>  Leave n ms between checks for changed work (default: 500)." << endl
			<< "    --no-precompute  Don't precompute the next epoch's DAG." << endl
#endif
			<< "Ethash verify mode:" << endl
			<< "    -w,--check-pow <headerHash> <seedHash> <difficulty> <nonce>  Check PoW credentials for validity." << endl
			<< endl
			<< "Benchmarking mode:" << endl
			<< "    -M,--benchmark  Benchmark for mining and exit; use with --cpu and --opencl." << endl
			<< "    --benchmark-warmup <seconds>  Set the duration of warmup for the benchmark tests (default: 3)." << endl
			<< "    --benchmark-trial <seconds>  Set the duration for each trial for the benchmark tests (default: 3)." << endl
			<< "    --benchmark-trials <n>  Set the duration of warmup for the benchmark tests (default: 5)." << endl
			<< "Simulation mode:" << endl
			<< "    -S,--simulation Mining test mode. Used to validate kernel optimizations." << endl
#if ETH_JSONRPC || !ETH_TRUE
			<< "    --phone-home <on/off>  When benchmarking, publish results (default: on)" << endl
#endif
			<< "DAG creation mode:" << endl
			<< "    -D,--create-dag <number>  Create the DAG in preparation for mining on given block and exit." << endl
			<< "Mining configuration:" << endl
			<< "    -C,--cpu  When mining, use the CPU." << endl
			<< "    -G,--opencl  When mining use the GPU via OpenCL." << endl
			<< "    -U,--cuda  When mining use the GPU via CUDA." << endl
			<< "    --opencl-platform <n>  When mining using -G/--opencl use OpenCL platform n (default: 0)." << endl
			<< "    --opencl-device <n>  When mining using -G/--opencl use OpenCL device n (default: 0)." << endl
			<< "    -t, --mining-threads <n> Limit number of CPU/GPU miners to n (default: use everything available on selected platform)" << endl
			<< "    --allow-opencl-cpu Allows CPU to be considered as an OpenCL device if the OpenCL platform supports it." << endl
			<< "    --list-devices List the detected OpenCL/CUDA devices and exit." << endl
			<< "    --current-block Let the miner know the current block number at configuration time. Will help determine DAG size and required GPU memory." << endl
#if ETH_ETHASHCL || !ETH_TRUE
			<< "    --cl-extragpu-mem Set the memory (in MB) you believe your GPU requires for stuff other than mining. Windows rendering e.t.c.." << endl
			<< "    --cl-local-work Set the OpenCL local work size. Default is " << toString(ethash_cl_miner::c_defaultLocalWorkSize) << endl
			<< "    --cl-global-work Set the OpenCL global work size as a multiple of the local work size. Default is " << toString(ethash_cl_miner::c_defaultGlobalWorkSizeMultiplier) << " * " << toString(ethash_cl_miner::c_defaultLocalWorkSize) << endl
			<< "    --cl-ms-per-batch Set the OpenCL target milliseconds per batch (global workgroup size). Default is " << toString(ethash_cl_miner::c_defaultMSPerBatch) << ". If 0 is given then no autoadjustment of global work size will happen" << endl
#endif
#if ETH_ETHASHCUDA || !ETH_TRUE
			<< "    --cuda-extragpu-mem Set the memory (in MB) you believe your GPU requires for stuff other than mining. Windows rendering e.t.c.." << endl
			<< "    --cuda-block-size Set the CUDA block work size. Default is " << toString(ethash_cuda_miner::c_defaultBlockSize) << endl
			<< "    --cuda-grid-size Set the CUDA grid size. Default is " << toString(ethash_cuda_miner::c_defaultGridSize) << endl
			<< "    --cuda-streams Set the number of CUDA streams. Default is " << toString(ethash_cuda_miner::c_defaultNumStreams) << endl
			<< "    --cuda-turbo Get a bit of extra hashrate at the cost of high CPU load... Default is false" << endl
			<< "    --cuda-devices <0 1 ..n> Select which GPUs to mine on. Default is to use all" << endl
#endif
			;
	}

	enum class MinerType
	{
		CPU,
		CL,
		CUDA
	};

	MinerType minerType() const { return m_minerType; }
	bool shouldPrecompute() const { return m_precompute; }

private:
	void doInitDAG(unsigned _n)
	{
		h256 seedHash = EthashAux::seedHash(_n);
		cout << "Initializing DAG for epoch beginning #" << (_n / 30000 * 30000) << " (seedhash " << seedHash.abridged() << "). This will take a while." << endl;
		EthashAux::full(seedHash, true);
		exit(0);
	}

	void doBenchmark(MinerType _m, bool _phoneHome, unsigned _warmupDuration = 15, unsigned _trialDuration = 3, unsigned _trials = 5)
	{
		Ethash::BlockHeader genesis;
		genesis.setDifficulty(1 << 18);
		cdebug << genesis.boundary();

		GenericFarm<EthashProofOfWork> f;
		map<string, GenericFarm<EthashProofOfWork>::SealerDescriptor> sealers;
		sealers["cpu"] = GenericFarm<EthashProofOfWork>::SealerDescriptor{&EthashCPUMiner::instances, [](GenericMiner<EthashProofOfWork>::ConstructionInfo ci){ return new EthashCPUMiner(ci); }};
#if ETH_ETHASHCL
		sealers["opencl"] = GenericFarm<EthashProofOfWork>::SealerDescriptor{&EthashGPUMiner::instances, [](GenericMiner<EthashProofOfWork>::ConstructionInfo ci){ return new EthashGPUMiner(ci); }};
#endif
#if ETH_ETHASHCUDA
		sealers["cuda"] = GenericFarm<EthashProofOfWork>::SealerDescriptor{ &EthashCUDAMiner::instances, [](GenericMiner<EthashProofOfWork>::ConstructionInfo ci){ return new EthashCUDAMiner(ci); } };
#endif
		f.setSealers(sealers);
		f.onSolutionFound([&](EthashProofOfWork::Solution) { return false; });

		string platformInfo = _m == MinerType::CPU ? "CPU" : (_m == MinerType::CL ? "CL" : "CUDA");
		cout << "Benchmarking on platform: " << platformInfo << endl;

		cout << "Preparing DAG..." << endl;
		genesis.prep();

		genesis.setDifficulty(u256(1) << 63);
		f.setWork(genesis);
		if (_m == MinerType::CPU)
			f.start("cpu");
		else if (_m == MinerType::CL)
			f.start("opencl");
		else if (_m == MinerType::CUDA)
			f.start("cuda");

		map<uint64_t, WorkingProgress> results;
		uint64_t mean = 0;
		uint64_t innerMean = 0;
		for (unsigned i = 0; i <= _trials; ++i)
		{
			if (!i)
				cout << "Warming up..." << endl;
			else
				cout << "Trial " << i << "... " << flush;
			this_thread::sleep_for(chrono::seconds(i ? _trialDuration : _warmupDuration));

			auto mp = f.miningProgress();
			f.resetMiningProgress();
			if (!i)
				continue;
			auto rate = mp.rate();

			cout << rate << endl;
			results[rate] = mp;
			mean += rate;
		}
		f.stop();
		int j = -1;
		for (auto const& r: results)
			if (++j > 0 && j < (int)_trials - 1)
				innerMean += r.second.rate();
		innerMean /= (_trials - 2);
		cout << "min/mean/max: " << results.begin()->second.rate() << "/" << (mean / _trials) << "/" << results.rbegin()->second.rate() << " H/s" << endl;
		cout << "inner mean: " << innerMean << " H/s" << endl;

		(void)_phoneHome;
#if ETH_JSONRPC || !ETH_TRUE
		if (_phoneHome)
		{
			cout << "Phoning home to find world ranking..." << endl;
			jsonrpc::HttpClient client("http://gav.ethdev.com:3000");
			PhoneHome rpc(client);
			try
			{
				unsigned ranking = rpc.report_benchmark(platformInfo, innerMean);
				cout << "Ranked: " << ranking << " of all benchmarks." << endl;
			}
			catch (...)
			{
			}
		}
#endif
		exit(0);
	}

	void doSimulation(MinerType _m, int difficulty = 20)
	{
		Ethash::BlockHeader genesis;
		genesis.setDifficulty(1 << 18);
		cdebug << genesis.boundary();

		GenericFarm<EthashProofOfWork> f;
		map<string, GenericFarm<EthashProofOfWork>::SealerDescriptor> sealers;
		sealers["cpu"] = GenericFarm<EthashProofOfWork>::SealerDescriptor{ &EthashCPUMiner::instances, [](GenericMiner<EthashProofOfWork>::ConstructionInfo ci){ return new EthashCPUMiner(ci); } };
#if ETH_ETHASHCL
		sealers["opencl"] = GenericFarm<EthashProofOfWork>::SealerDescriptor{ &EthashGPUMiner::instances, [](GenericMiner<EthashProofOfWork>::ConstructionInfo ci){ return new EthashGPUMiner(ci); } };
#endif
#if ETH_ETHASHCUDA
		sealers["cuda"] = GenericFarm<EthashProofOfWork>::SealerDescriptor{ &EthashCUDAMiner::instances, [](GenericMiner<EthashProofOfWork>::ConstructionInfo ci){ return new EthashCUDAMiner(ci); } };
#endif
		f.setSealers(sealers);

		string platformInfo = _m == MinerType::CPU ? "CPU" : (_m == MinerType::CL ? "CL" : "CUDA");
		cout << "Benchmarking on platform: " << platformInfo << endl;

		cout << "Preparing DAG..." << endl;
		genesis.prep();

		genesis.setDifficulty(u256(1) << difficulty);
		f.setWork(genesis);

		if (_m == MinerType::CPU)
			f.start("cpu");
		else if (_m == MinerType::CL)
			f.start("opencl");
		else if (_m == MinerType::CUDA)
			f.start("cuda");

		EthashAux::FullType dag;

		int time = 0;

		EthashProofOfWork::WorkPackage current = EthashProofOfWork::WorkPackage(genesis);
		while (true) {
			bool completed = false;
			EthashProofOfWork::Solution solution;
			f.onSolutionFound([&](EthashProofOfWork::Solution sol)
			{
				solution = sol;
				return completed = true;
			});
			for (unsigned i = 0; !completed; ++i)
			{
				auto mp = f.miningProgress();
				f.resetMiningProgress();

				cnote << "Mining on difficulty " << difficulty << " " << mp;
				this_thread::sleep_for(chrono::milliseconds(1000));
				time++;
			}
			//cnote << "Solution found";
			cnote << "Difficulty:" << difficulty << "  Nonce:" << solution.nonce.hex();
			//cnote << "  Mixhash:" << solution.mixHash.hex();
			//cnote << "  Header-hash:" << current.headerHash.hex();
			//cnote << "  Seedhash:" << current.seedHash.hex();
			//cnote << "  Target: " << h256(current.boundary).hex();
			//cnote << "  Ethash: " << h256(EthashAux::eval(current.seedHash, current.headerHash, solution.nonce).value).hex();
			if (EthashAux::eval(current.seedHash, current.headerHash, solution.nonce).value < current.boundary)
			{
				cnote << "SUCCESS: GPU gave correct result!";
			}
			else
				cwarn << "FAILURE: GPU gave incorrect result!";

			if (time < 12)
				difficulty++;
			else if (time > 18)
				difficulty--;
			time = 0;
			genesis.setDifficulty(u256(1) << difficulty);
			genesis.noteDirty();
			//f.setWork(genesis);
		
			h256 hh;
			std::random_device engine;
			hh.randomize(engine);
			h256 newSeedHash = h256();

			current.headerHash = hh;
			current.seedHash = newSeedHash;
			current.boundary = genesis.boundary();
			minelog << "Generated random work package:";
			minelog << "  Header-hash:" << current.headerHash.hex();
			minelog << "  Seedhash:" << current.seedHash.hex();
			minelog << "  Target: " << h256(current.boundary).hex();
			f.setWork(current);
			

			//current = EthashProofOfWork::WorkPackage(genesis);
		}
	}

	void doFarm(MinerType _m, string const& _remote, unsigned _recheckPeriod)
	{
		map<string, GenericFarm<EthashProofOfWork>::SealerDescriptor> sealers;
		sealers["cpu"] = GenericFarm<EthashProofOfWork>::SealerDescriptor{&EthashCPUMiner::instances, [](GenericMiner<EthashProofOfWork>::ConstructionInfo ci){ return new EthashCPUMiner(ci); }};
#if ETH_ETHASHCL
		sealers["opencl"] = GenericFarm<EthashProofOfWork>::SealerDescriptor{&EthashGPUMiner::instances, [](GenericMiner<EthashProofOfWork>::ConstructionInfo ci){ return new EthashGPUMiner(ci); }};
#endif
#if ETH_ETHASHCUDA
		sealers["cuda"] = GenericFarm<EthashProofOfWork>::SealerDescriptor{ &EthashCUDAMiner::instances, [](GenericMiner<EthashProofOfWork>::ConstructionInfo ci){ return new EthashCUDAMiner(ci); } };
#endif
		(void)_m;
		(void)_remote;
		(void)_recheckPeriod;
#if ETH_JSONRPC || !ETH_TRUE
		jsonrpc::HttpClient client(_remote);

		h256 id = h256::random();
		::FarmClient rpc(client);
		GenericFarm<EthashProofOfWork> f;
		f.setSealers(sealers);
		if (_m == MinerType::CPU)
			f.start("cpu");
		else if (_m == MinerType::CL)
			f.start("opencl");
		else if (_m == MinerType::CUDA)
			f.start("cuda");
		EthashProofOfWork::WorkPackage current;
		EthashAux::FullType dag;
		while (true)
			try
			{
				bool completed = false;
				EthashProofOfWork::Solution solution;
				f.onSolutionFound([&](EthashProofOfWork::Solution sol)
				{
					solution = sol;
					return completed = true;
				});
				for (unsigned i = 0; !completed; ++i)
				{
					auto mp = f.miningProgress();
					f.resetMiningProgress();
					if (current)
						minelog << "Mining on PoWhash" << current.headerHash << ": " << mp;
					else
						minelog << "Getting work package...";

					auto rate = mp.rate();

					try
					{
						rpc.eth_submitHashrate(toJS((u256)rate), "0x" + id.hex());
					}
					catch (jsonrpc::JsonRpcException const& _e)
					{
						cwarn << "Failed to submit hashrate.";
						cwarn << boost::diagnostic_information(_e);
					}

					Json::Value v = rpc.eth_getWork();
					h256 hh(v[0].asString());
					h256 newSeedHash(v[1].asString());
					if (current.seedHash != newSeedHash)
						minelog << "Grabbing DAG for" << newSeedHash;
					if (!(dag = EthashAux::full(newSeedHash, true, [&](unsigned _pc){ cout << "\rCreating DAG. " << _pc << "% done..." << flush; return 0; })))
						BOOST_THROW_EXCEPTION(DAGCreationFailure());
					if (m_precompute)
						EthashAux::computeFull(sha3(newSeedHash), true);
					if (hh != current.headerHash)
					{
						current.headerHash = hh;
						current.seedHash = newSeedHash;
						current.boundary = h256(fromHex(v[2].asString()), h256::AlignRight);
						minelog << "Got work package:";
						minelog << "  Header-hash:" << current.headerHash.hex();
						minelog << "  Seedhash:" << current.seedHash.hex();
						minelog << "  Target: " << h256(current.boundary).hex();
						f.setWork(current);
					}
					this_thread::sleep_for(chrono::milliseconds(_recheckPeriod));
				}
				cnote << "Solution found; Submitting to" << _remote << "...";
				cnote << "  Nonce:" << solution.nonce.hex();
				cnote << "  Mixhash:" << solution.mixHash.hex();
				cnote << "  Header-hash:" << current.headerHash.hex();
				cnote << "  Seedhash:" << current.seedHash.hex();
				cnote << "  Target: " << h256(current.boundary).hex();
				cnote << "  Ethash: " << h256(EthashAux::eval(current.seedHash, current.headerHash, solution.nonce).value).hex();
				if (EthashAux::eval(current.seedHash, current.headerHash, solution.nonce).value < current.boundary)
				{
					bool ok = rpc.eth_submitWork("0x" + toString(solution.nonce), "0x" + toString(current.headerHash), "0x" + toString(solution.mixHash));
					if (ok)
						cnote << "B-) Submitted and accepted.";
					else
						cwarn << ":-( Not accepted.";
				}
				else
					cwarn << "FAILURE: GPU gave incorrect result!";
				current.reset();
			}
			catch (jsonrpc::JsonRpcException&)
			{
				for (auto i = 3; --i; this_thread::sleep_for(chrono::seconds(1)))
					cerr << "JSON-RPC problem. Probably couldn't connect. Retrying in " << i << "... \r";
				cerr << endl;
			}
#endif
		exit(0);
	}

	/// Operating mode.
	OperationMode mode;

	/// Mining options
	MinerType m_minerType = MinerType::CPU;
	unsigned m_openclPlatform = 0;
	unsigned m_openclDevice = 0;
	unsigned m_miningThreads = UINT_MAX;
	bool m_shouldListDevices = false;
	bool m_clAllowCPU = false;
#if ETH_ETHASHCL || !ETH_TRUE
#if !ETH_ETHASHCUDA || !ETH_TRUE
	unsigned m_globalWorkSizeMultiplier = ethash_cl_miner::c_defaultGlobalWorkSizeMultiplier;
	unsigned m_localWorkSize = ethash_cl_miner::c_defaultLocalWorkSize;
#endif
	unsigned m_msPerBatch = ethash_cl_miner::c_defaultMSPerBatch;
#endif
#if ETH_ETHASHCUDA || !ETH_TRUE
	unsigned m_globalWorkSizeMultiplier = ethash_cuda_miner::c_defaultGridSize;
	unsigned m_localWorkSize = ethash_cuda_miner::c_defaultBlockSize;
	unsigned m_cudaDeviceCount = 0;
	unsigned m_cudaDevices[16];
	unsigned m_numStreams = ethash_cuda_miner::c_defaultNumStreams;
	bool m_cudaHighCPULoad = false;
#endif
	uint64_t m_currentBlock = 0;
	// default value is 350MB of GPU memory for other stuff (windows system rendering, e.t.c.)
	unsigned m_extraGPUMemory = 350000000;

	/// DAG initialisation param.
	unsigned m_initDAG = 0;

	/// Benchmarking params
	bool m_phoneHome = true;
	unsigned m_benchmarkWarmup = 3;
	unsigned m_benchmarkTrial = 3;
	unsigned m_benchmarkTrials = 5;

	/// Farm params
	string m_farmURL = "http://127.0.0.1:8545";
	unsigned m_farmRecheckPeriod = 500;
#if defined (_WIN32) && defined(__MACHINEX86)
	bool m_precompute = false;
#else
	bool m_precompute = true;
#endif
};
