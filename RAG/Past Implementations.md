## [Song 2025](../../Papers/2025/Song2025RAGAgentFaultPowerSystems.pdf)
* Uses LLM and structured database (typical) [Source](Papers/2025/Song2025RAGAgentFaultPowerSystems.pdf#page=1&selection=185,39,193,61&color=yellow)
* Integrated routing (via intention identification) [Source](Papers/2025/Song2025RAGAgentFaultPowerSystems.pdf#page=1&selection=194,0,196,52&color=yellow)
	* Uses prompt chaining (query normalization and evidence based reasoning)
	* Uses tool invocation (SQL based aggregation functions)
* Multi-level routing system
	1. Intention identification (distinguishes between cause inferences of cases or summarization of failure records), decides between DeepSeek R1 or DeepSeek V3 [Source](Papers/2025/Song2025RAGAgentFaultPowerSystems.pdf#page=2&selection=97,0,106,30&color=yellow)
	2. Second stage intention identification (classifies types of statistical summarization queries) [Source](Papers/2025/Song2025RAGAgentFaultPowerSystems.pdf#page=2&selection=108,0,139,54&color=yellow)
* Vector embeddings included documents from heterogenous sources.