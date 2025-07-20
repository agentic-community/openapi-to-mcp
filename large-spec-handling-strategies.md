# Large OpenAPI Specification Handling Strategies

## Current Limitations

The openapi-to-mcp tool currently processes OpenAPI specifications as monolithic units, which creates limitations when dealing with large specs that exceed Claude's context window. The current implementation:

- Loads entire specs into memory at once
- Sends complete specs to LLM in single requests
- Has no size validation or chunking logic
- Risks truncation or failure with large specifications
- Processes everything via single evaluation templates

## Recommended Approaches

### 1. Path-Based Batch Processing (Primary Recommendation)

**Strategy**: Process OpenAPI specs in batches of 3-5 paths, including all their referenced schemas and shared context (info, servers, security).

**Implementation**:
- Group paths into small batches (3-5 paths per batch)
- Extract all schema dependencies for each batch
- Include shared context (info, servers, security) in each chunk
- Process batches independently and aggregate results

**Benefits**:
- Maintains complete context for proper evaluation
- Optimal balance between chunk size and context completeness
- Enables parallel processing while preserving semantic coherence
- Handles large specs effectively (tested with OpenAI's 1.9MB spec)

**Code Structure**:
```python
# Process paths in batches
batch_size = 3  # 3-5 paths per batch recommended
for i in range(0, len(paths), batch_size):
    batch_paths = paths[i:i+batch_size]
    
    # Collect all schemas referenced by this batch
    referenced_schemas = extract_schema_refs(batch_paths)
    
    # Build chunk with complete context
    chunk = {
        'info': spec['info'],
        'servers': spec['servers'],
        'security': spec.get('security', []),
        'paths': {path: spec['paths'][path] for path in batch_paths},
        'components': {
            'schemas': {name: spec['components']['schemas'][name] 
                       for name in referenced_schemas},
            'securitySchemes': spec['components'].get('securitySchemes', {})
        }
    }
    
    # Evaluate batch
    batch_results = await evaluate_chunk(chunk)
    all_results.extend(batch_results)

# Final aggregation
final_report = aggregate_results(all_results)
```

**Real-World Example**:
Based on analysis of OpenAI's spec (109 paths, 571 schemas):
- Each path can reference 5-10+ schemas
- Single path definitions can be 40KB+
- Batching 3-5 paths keeps chunks well within context limits
- Preserves all necessary relationships for accurate evaluation

### 2. Schema-First Chunking

**Strategy**: Process component schemas separately before analyzing operations.

**Implementation**:
- First pass: Analyze all schemas in `components/schemas`
- Second pass: Process operations with schema context
- Build type understanding before operation analysis

**Benefits**:
- Better type comprehension
- Reduces redundant schema analysis
- Enables schema-aware operation evaluation

### 3. Size-Based Adaptive Chunking

**Strategy**: Dynamically group content based on estimated token count.

**Implementation**:
- Use libraries like `tiktoken` for accurate token estimation
- Group operations to fit within context window limits
- Adjust chunk size based on complexity

**Benefits**:
- Optimal context window utilization
- Handles varying operation complexity
- Prevents token limit overruns

### 4. Hierarchical Processing

**Strategy**: Multi-pass analysis with increasing detail levels.

**Processing Levels**:
1. **High-level**: Spec metadata (info, servers, security)
2. **Schema analysis**: Component definitions and relationships
3. **Operation analysis**: Detailed endpoint evaluation
4. **Cross-cutting**: Relationships and consistency checks

**Benefits**:
- Progressive context building
- Handles complex interdependencies
- Enables early validation

### 5. Streaming/Progressive Analysis

**Strategy**: Incremental processing with context accumulation.

**Implementation**:
- Process spec sections in dependency order
- Use intermediate results to inform later chunks
- Build comprehensive understanding progressively

**Benefits**:
- Memory efficient
- Handles very large specifications
- Maintains context between chunks

### 6. Smart Filtering

**Strategy**: Pre-filter and prioritize spec content.

**Filtering Criteria**:
- HTTP methods (focus on complex operations)
- Tags or categories
- Operation complexity scores
- Business criticality

**Benefits**:
- Focuses on important operations first
- Reduces processing time
- Enables tiered analysis

## Implementation Considerations

### Token Management

```python
# Constants to add to config
MAX_CHUNK_TOKENS: int = 8000
MAX_SPEC_SIZE_CHARS: int = 100000
OVERLAP_TOKENS: int = 500  # For context continuity
```

### Chunking Configuration

```yaml
# Addition to config.yml
chunking:
  enabled: true
  strategy: "operation_based"  # operation_based, size_based, schema_first
  max_chunk_tokens: 8000
  overlap_tokens: 500
  parallel_processing: true
  max_concurrent_chunks: 5
```

### Error Handling

- Fallback to smaller chunks on failures
- Progressive detail reduction for oversized operations
- Graceful degradation for context limit hits

### Result Aggregation

- Consistent output format across chunks
- Conflict resolution for overlapping analysis
- Comprehensive final summarization
- Relationship preservation between chunks

## Recommended Implementation Order

1. **Phase 1**: Implement path-based batch processing (3-5 paths per batch)
2. **Phase 2**: Add schema dependency extraction and inclusion
3. **Phase 3**: Implement parallel batch processing
4. **Phase 4**: Add size estimation and adaptive batch sizing
5. **Phase 5**: Implement result aggregation and final summarization

## Tools and Libraries

- **Token Estimation**: `tiktoken` for accurate token counting
- **Spec Parsing**: Enhanced error handling in existing YAML/JSON parsers
- **Parallel Processing**: `asyncio` for concurrent chunk processing
- **Configuration**: Extended `config.yml` for chunking parameters

## Expected Benefits

- Handle arbitrarily large OpenAPI specifications
- Improved processing speed through parallel analysis
- Better analysis quality through focused evaluation
- Reduced memory usage
- Enhanced error resilience
- Scalable architecture for future enhancements

## Migration Strategy

The chunking implementation should be backward compatible:
- Default to monolithic processing for small specs
- Automatic chunking activation based on size thresholds
- Configuration options to force chunking behavior
- Gradual rollout with feature flags