Graph Profiling
---------------

You can enable per-pipeline graph profiling with Vulkan® timestamp queries.
Profiling covers TOSA graph operators, MotionEngine graph operators, and
optical-flow compute pipelines. Profiling is disabled by default. When enabled,
graph command-buffer submits remain asynchronous; timestamp results are
collected when the application waits on fences, waits for a queue or device to
become idle, or when the profiling property is queried. Profiling results are
saved only as a queryable data graph pipeline property.

.. code-block:: shell

    export VMEL_GRAPH_PROFILING=1

The profiling property returns JSON with a ``samples`` array containing one
entry per profiled internal compute dispatch, including ``pipeline_kind``,
``operator_name``, raw cycle counts, and ``time_ms``, plus a ``by_operator``
summary with total, average, minimum, and maximum time per profiled pipeline.
