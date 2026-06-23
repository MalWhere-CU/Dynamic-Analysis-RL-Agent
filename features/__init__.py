"""
Feature extraction and state construction for the MalWhere RL environment.

Source: train/train.ipynb, Cell 4 (lines 277-609) and Cell 5 (lines 650-793)
Status: Copied unchanged from notebook.
Why: Feature extraction and normalization must match training exactly to produce
     consistent 35-dimensional state vectors for the trained model.
"""

from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

from agent import Action


class CAPEFeatureExtractor:
    """
    Comprehensive feature extraction system for CAPE v2 malware analysis reports.

    This class implements hierarchical feature extraction corresponding to the
    reinforcement learning action space, enabling progressive information revelation.

    Design Pattern: Static methods allow stateless operation for parallel processing.
    """

    # Domain Knowledge: API calls indicative of code injection techniques
    INJECTION_APIS = {
        'WriteProcessMemory', 'VirtualAllocEx', 'CreateRemoteThread',
        'NtWriteVirtualMemory', 'RtlCreateUserThread', 'QueueUserAPC'
    }

    @staticmethod
    def safe_get(data: Dict, keys: List[str], default=0):
        """
        Safely navigate nested dictionary structures with fallback defaults.

        Args:
            data: Source dictionary
            keys: List of keys representing nested path
            default: Default value if path doesn't exist

        Returns:
            Retrieved value or default
        """
        current = data
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current

    @classmethod
    def extract_basic_features(cls, report: Dict) -> Dict[str, float]:
        """
        Extract baseline features available without expensive operations (Action 0).

        Feature Categories:
        - File metadata: Size, type, CAPE classification code
        - Process information: Count, threads, tree complexity
        - API call statistics: Total invocations

        Args:
            report: CAPE v2 JSON report

        Returns:
            Dictionary of normalized feature values
        """
        features = {}

        # Extract file metadata from target section
        target = report.get('target', {})
        file_info = target.get('file', {})

        features['file_size'] = float(file_info.get('size', 0))
        features['cape_type_code'] = float(file_info.get('cape_type_code', 0))

        # Encode file type as ordinal variable
        file_type = str(file_info.get('type', '')).lower()
        if 'pe32' in file_type or 'exe' in file_type:
            features['file_type_ord'] = 1.0
        elif 'dll' in file_type:
            features['file_type_ord'] = 2.0
        else:
            features['file_type_ord'] = 0.0

        # Behavioral indicators from process monitoring
        behavior = report.get('behavior', {})
        processes = behavior.get('processes', [])

        features['n_processes'] = float(len(processes))

        # Aggregate thread count across all processes
        thread_count = 0
        for proc in processes:
            threads = proc.get('threads', [])
            thread_count += len(threads)
        features['n_threads_total'] = float(thread_count)

        # Process tree complexity metric
        processtree = behavior.get('processtree', [])
        features['proc_tree_nodes'] = float(cls._count_tree_nodes(processtree))

        # API invocation frequency
        total_api_calls = 0
        for proc in processes:
            calls = proc.get('calls', [])
            total_api_calls += len(calls)
        features['total_api_calls'] = float(total_api_calls)

        return features

    @classmethod
    def extract_memory_features(cls, report: Dict) -> Dict[str, float]:
        """
        Extract memory-related behavioral indicators (Action 1).

        Feature Categories:
        - Enhanced memory events from CAPE monitoring
        - Code injection pattern detection via API analysis

        Args:
            report: CAPE v2 JSON report

        Returns:
            Dictionary of memory-specific features
        """
        features = {}
        behavior = report.get('behavior', {})
        processes = behavior.get('processes', [])

        # Enhanced monitoring events
        enhanced = behavior.get('enhanced', [])
        features['n_enhanced_events'] = float(len(enhanced))

        # Injection detection: Count suspicious API invocations
        injection_count = 0
        for proc in processes:
            calls = proc.get('calls', [])
            for call in calls:
                api = call.get('api', '')
                if any(inj_api in api for inj_api in cls.INJECTION_APIS):
                    injection_count += 1

        features['n_injection_indicators'] = float(injection_count)

        return features

    @classmethod
    def extract_filesystem_features(cls, report: Dict) -> Dict[str, float]:
        """
        Extract filesystem interaction patterns (Action 2).

        Feature Categories:
        - File operations: Read, write, delete counts
        - Dropped artifacts: Count and cumulative size

        Args:
            report: CAPE v2 JSON report

        Returns:
            Dictionary of filesystem features
        """
        features = {}
        behavior = report.get('behavior', {})
        summary = behavior.get('summary', {})

        # File operation statistics from behavior summary
        features['files_total'] = float(len(summary.get('files', [])))
        features['read_files'] = float(len(summary.get('read_files', [])))
        features['write_files'] = float(len(summary.get('write_files', [])))
        features['delete_files'] = float(len(summary.get('delete_files', [])))

        # Dropped file analysis
        dropped = report.get('dropped', [])
        features['n_dropped'] = float(len(dropped))
        dropped_sizes = [d.get('size', 0) for d in dropped]
        features['dropped_total_size'] = float(sum(dropped_sizes))

        return features

    @classmethod
    def extract_network_features(cls, report: Dict) -> Dict[str, float]:
        """
        Extract network activity patterns from API call analysis (Action 3).

        Feature Engineering Strategy:
        Instead of relying on CAPE's network section (which may be incomplete),
        this method analyzes API calls to identify network-related behavior.

        Feature Categories:
        - API call classification: DNS, HTTP, socket operations
        - Network activity ratios: Relative frequency of network operation types
        - Signature-based indicators: Network-related CAPE signatures

        Args:
            report: CAPE v2 JSON report

        Returns:
            Dictionary of network behavior features
        """
        features = {}
        behavior = report.get('behavior', {})
        processes = behavior.get('processes', [])

        # Initialize counters for different network operation types
        total_network_calls = 0
        network_calls_by_type = defaultdict(int)

        # API categorization based on Windows networking functions
        dns_apis = ['dns', 'gethost', 'getaddrinfo', 'getnameinfo']
        http_apis = ['http', 'internet', 'winhttp', 'url']
        socket_apis = ['socket', 'connect', 'bind', 'listen', 'accept',
                        'send', 'recv', 'closesocket', 'wsa']
        crypto_url_apis = ['cryptretrieveobjectbyurl', 'urlcanonicalize']
        network_apis = ['ras', 'getadaptersaddresses', 'setsockopt',
                        'ioctlsocket', 'wsa', 'getsockopt']

        # Iterate through all API calls and classify network operations
        for proc in processes:
            calls = proc.get('calls', [])
            for call in calls:
                api = call.get('api', '').lower()
                category = call.get('category', '').lower()

                is_network_call = False

                # Classification by CAPE category
                if category == 'network':
                    is_network_call = True

                # Classification by API function name pattern matching
                elif any(net_api in api for net_api in dns_apis):
                    network_calls_by_type['dns'] += 1
                    is_network_call = True
                elif any(net_api in api for net_api in http_apis):
                    network_calls_by_type['http'] += 1
                    is_network_call = True
                elif any(net_api in api for net_api in socket_apis):
                    network_calls_by_type['socket'] += 1
                    is_network_call = True
                elif any(net_api in api for net_api in crypto_url_apis):
                    network_calls_by_type['crypto_url'] += 1
                    is_network_call = True
                elif any(net_api in api for net_api in network_apis):
                    network_calls_by_type['other_network'] += 1
                    is_network_call = True

                if is_network_call:
                    total_network_calls += 1

        # Absolute frequency features
        features['total_network_calls'] = float(total_network_calls)
        features['dns_calls'] = float(network_calls_by_type.get('dns', 0))
        features['http_calls'] = float(network_calls_by_type.get('http', 0))
        features['socket_calls'] = float(network_calls_by_type.get('socket', 0))
        features['crypto_url_calls'] = float(network_calls_by_type.get('crypto_url', 0))
        features['other_network_calls'] = float(network_calls_by_type.get('other_network', 0))

        # Relative frequency features (normalized by total network activity)
        if total_network_calls > 0:
            features['dns_ratio'] = features['dns_calls'] / total_network_calls
            features['http_ratio'] = features['http_calls'] / total_network_calls
            features['socket_ratio'] = features['socket_calls'] / total_network_calls
            features['crypto_url_ratio'] = features['crypto_url_calls'] / total_network_calls
        else:
            features['dns_ratio'] = 0.0
            features['http_ratio'] = 0.0
            features['socket_ratio'] = 0.0
            features['crypto_url_ratio'] = 0.0

        # Signature-based network behavior indicators
        network_signatures = 0
        signatures = report.get('signatures', [])
        for sig in signatures:
            name = sig.get('name', '').lower()
            if any(net_term in name for net_term in
                    ['network', 'dns', 'http', 'c2', 'botnet', 'communication',
                    'download', 'upload', 'socket', 'connection']):
                network_signatures += 1

        features['network_signatures'] = float(network_signatures)

        return features

    @classmethod
    def extract_memory_dump_features(cls, report: Dict) -> Dict[str, float]:
        """
        Extract deep behavioral analysis features requiring memory dumps (Action 4).

        This represents the most computationally expensive analysis level,
        providing detailed insights into malware behavior.

        Feature Categories:
        - Behavioral anomalies detected by CAPE
        - Encrypted buffer analysis
        - CAPE signature matches and alert levels
        - Extracted payload characteristics

        Args:
            report: CAPE v2 JSON report

        Returns:
            Dictionary of deep analysis features
        """
        features = {}
        behavior = report.get('behavior', {})

        # Behavioral anomaly detection
        anomalies = behavior.get('anomaly', [])
        features['n_anomalies'] = float(len(anomalies))

        # Encrypted buffer detection (potential crypter/packer indicators)
        encrypted = behavior.get('encryptedbuffers', [])
        features['n_encryptedbuffers'] = float(len(encrypted))

        # CAPE signature analysis
        signatures = report.get('signatures', [])
        alert_signatures = [s for s in signatures if s.get('alert', False)]
        features['n_signatures'] = float(len(signatures))
        features['signatures_alert_count'] = float(len(alert_signatures))

        # Payload extraction analysis
        cape_section = report.get('CAPE', {})
        payloads = cape_section.get('payloads', [])
        features['n_payloads'] = float(len(payloads))
        payload_sizes = [p.get('size', 0) for p in payloads]
        features['payloads_total_size'] = float(sum(payload_sizes))

        return features

    @staticmethod
    def _count_tree_nodes(tree: List) -> int:
        """
        Recursively compute process tree complexity metric.

        Args:
            tree: Process tree structure from CAPE report

        Returns:
            Total node count in process tree
        """
        count = 0
        for node in tree:
            count += 1
            count += CAPEFeatureExtractor._count_tree_nodes(node.get('children', []))
        return count


class StateBuilder:
    """
    State representation builder for reinforcement learning environment.

    Architecture: Slot-based design where each action type has pre-allocated dimensions.
    This allows the neural network to learn which actions revealed which information.

    State Vector Structure:
    [Action_0_features | Action_1_features | ... | Action_4_features | metadata]

    Unrevealed action slots are zero-padded, creating a sparse representation that
    the network learns to interpret as missing information.
    """

    # Feature dimensionality for each action (empirically determined from extractor output)
    FEATURE_DIMS = {
        Action.CONTINUE: 6,           # Basic metadata and process counts
        Action.FOCUS_MEMORY: 3,       # Memory events and injection indicators
        Action.FOCUS_FILESYSTEM: 6,   # File operations and dropped artifacts
        Action.FOCUS_NETWORK: 11,     # Network API analysis and signatures
        Action.MEMORY_DUMP: 7,        # Deep behavioral analysis
    }

    TOTAL_FEATURE_DIM = sum(FEATURE_DIMS.values()) + 2  # +2 for temporal metadata (step_id, last_action)

    @classmethod
    def build_state(cls,
                    report: Dict,
                    revealed_actions: set,
                    step_id: int,
                    last_action: Optional[int] = None) -> np.ndarray:
        """
        Construct state vector from CAPE report based on revealed information.

        The state vector encodes both the extracted features and which analyses
        have been performed, enabling the agent to reason about information gaps.

        Args:
            report: CAPE v2 JSON report
            revealed_actions: Set of actions already taken (features available)
            step_id: Current timestep in episode
            last_action: Previous action taken (None for initial state)

        Returns:
            numpy.ndarray: Normalized state vector of dimension TOTAL_FEATURE_DIM
        """
        extractor = CAPEFeatureExtractor
        state_parts = []

        # Slot 0: Basic features (always revealed at episode start)
        feat0 = extractor.extract_basic_features(report)
        state_parts.extend(cls._normalize_features(feat0, Action.CONTINUE))

        # Slot 1: Memory features (revealed if Action 1 taken)
        if Action.FOCUS_MEMORY in revealed_actions:
            feat1 = extractor.extract_memory_features(report)
            state_parts.extend(cls._normalize_features(feat1, Action.FOCUS_MEMORY))
        else:
            state_parts.extend([0.0] * cls.FEATURE_DIMS[Action.FOCUS_MEMORY])

        # Slot 2: Filesystem features (revealed if Action 2 taken)
        if Action.FOCUS_FILESYSTEM in revealed_actions:
            feat2 = extractor.extract_filesystem_features(report)
            state_parts.extend(cls._normalize_features(feat2, Action.FOCUS_FILESYSTEM))
        else:
            state_parts.extend([0.0] * cls.FEATURE_DIMS[Action.FOCUS_FILESYSTEM])

        # Slot 3: Network features (revealed if Action 3 taken)
        if Action.FOCUS_NETWORK in revealed_actions:
            feat3 = extractor.extract_network_features(report)
            state_parts.extend(cls._normalize_features(feat3, Action.FOCUS_NETWORK))
        else:
            state_parts.extend([0.0] * cls.FEATURE_DIMS[Action.FOCUS_NETWORK])

        # Slot 4: Memory dump features (revealed if Action 4 taken)
        if Action.MEMORY_DUMP in revealed_actions:
            feat4 = extractor.extract_memory_dump_features(report)
            state_parts.extend(cls._normalize_features(feat4, Action.MEMORY_DUMP))
        else:
            state_parts.extend([0.0] * cls.FEATURE_DIMS[Action.MEMORY_DUMP])

        # Temporal metadata (normalized to [0,1] range)
        state_parts.append(float(step_id) / 20.0)  # Normalized step count (max 20 steps assumed)
        state_parts.append(float(last_action if last_action is not None else -1) / 10.0)

        return np.array(state_parts, dtype=np.float32)

    @staticmethod
    def _normalize_features(features: Dict[str, float], action: Action) -> List[float]:
        """
        Apply feature-specific normalization strategies.

        Different feature types require different normalization approaches:
        - Size/count features: Logarithmic scaling to handle wide dynamic range
        - Binary indicators: Direct scaling
        - Categorical features: Ordinal encoding

        Args:
            features: Raw feature dictionary
            action: Action type (determines expected dimension)

        Returns:
            List of normalized feature values
        """
        normalized = []

        # Feature-specific normalization strategies
        for key, value in features.items():
            if 'size' in key or 'total' in key:
                # Logarithmic normalization for size-based features (handles exponential distributions)
                if value > 0:
                    normalized.append(np.log1p(value) / 15.0)  # log(1+x)/15 scales large values
                else:
                    normalized.append(0.0)
            elif 'count' in key or 'n_' in key:
                # Linear normalization with saturation for count features
                normalized.append(min(value / 100.0, 1.0))
            else:
                # Default normalization for categorical and other features
                normalized.append(min(value / 10.0, 1.0))

        # Dimension alignment: Pad or truncate to match expected slot size
        expected_dim = StateBuilder.FEATURE_DIMS[action]
        if len(normalized) < expected_dim:
            normalized.extend([0.0] * (expected_dim - len(normalized)))
        elif len(normalized) > expected_dim:
            normalized = normalized[:expected_dim]

        return normalized
