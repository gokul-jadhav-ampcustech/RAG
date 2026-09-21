"""MCP probe — discovers tools and tests one call."""
import asyncio
import sys

async def main():
    from app.services.mcp_client import _async_list_tools, _async_call_tool
    from app.config import get_settings
    s = get_settings()
    print(f"MCP_ENABLED : {s.mcp_enabled}")
    print(f"MCP_URL     : {s.mcp_server_url}")
    print(f"TOKEN SET   : {bool(s.mcp_api_token.strip())}")
    print()

    print("=== Discovering tools ===")
    try:
        tools = await _async_list_tools()
        print(f"Total tools: {len(tools)}")
        for t in tools:
            schema = t.get("input_schema", {})
            props  = list(schema.get("properties", {}).keys())
            req    = schema.get("required", [])
            print(f"  {t['name']}")
            print(f"    desc   : {t['description']}")
            print(f"    params : {props}")
            print(f"    required: {req}")
    except Exception as e:
        print(f"DISCOVERY ERROR: {type(e).__name__}: {e}")
        return

    if not tools:
        print("No tools found.")
        return

    # Quick smoke-test: call the first tool that needs no required params
    for t in tools:
        schema   = t.get("input_schema", {})
        required = schema.get("required", [])
        if not required:
            print(f"\n=== Smoke-test: calling '{t['name']}' (no required params) ===")
            try:
                result = await _async_call_tool(t["name"], {})
                print(f"RESULT: {result[:300]}")
            except Exception as e:
                print(f"CALL ERROR: {type(e).__name__}: {e}")
            break

asyncio.run(main())
