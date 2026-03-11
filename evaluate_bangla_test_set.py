import json
import requests
import time
from pathlib import Path
from collections import defaultdict

# Configuration
TEST_SET_FILE = r"C:\Users\Administrator\Downloads\wget installation (1)\wget installation\bangla_test_set_500.json"
RESULTS_FILE = r"C:\Users\Administrator\Downloads\wget installation (1)\wget installation\bangla_test_results.json"
REPORT_FILE = r"C:\Users\Administrator\Downloads\wget installation (1)\wget installation\bangla_test_report.txt"

# API Configuration
APIS = [
    {
        "name": "Qdrant",
        "url": "http://localhost:8081/hybrid_search",
        "port": 8081
    },
    {
        "name": "FAISS + pgvector",
        "url": "http://localhost:8082/hybrid_search",
        "port": 8082
    },
    {
        "name": "pgvector Only",
        "url": "http://localhost:8083/hybrid_search",
        "port": 8083
    }
]

# Search parameters
TOP_K = 5
SEMANTIC_WEIGHT = 0.7
KEYWORD_WEIGHT = 0.3

def load_test_set(filepath):
    """Load test set from JSON"""
    print(f"📂 Loading test set from: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        test_set = json.load(f)
    print(f"✅ Loaded {len(test_set)} test queries")
    return test_set

def query_api(api, query_text, top_k=5, semantic_weight=0.7, keyword_weight=0.3):
    """Query a single API"""
    try:
        payload = {
            "query": query_text,
            "top_k": top_k,
            "semantic_weight": semantic_weight,
            "keyword_weight": keyword_weight
        }
        
        response = requests.post(api["url"], json=payload, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"HTTP {response.status_code}"}
    
    except requests.exceptions.Timeout:
        return {"error": "Timeout"}
    except requests.exceptions.ConnectionError:
        return {"error": "Connection refused"}
    except Exception as e:
        return {"error": str(e)}

def normalize_filename(filename):
    """Normalize filename for comparison"""
    # Remove extensions and convert to lowercase
    normalized = filename.replace('_djvu.txt', '').replace('.txt', '').lower()
    return normalized

def check_success(expected_file, results, top_n=1):
    """Check if expected file is in top N results"""
    if "error" in results or "results" not in results:
        return False, None, None
    
    expected_normalized = normalize_filename(expected_file)
    
    for i, result in enumerate(results["results"][:top_n]):
        result_file = result.get("source_file", "")
        result_normalized = normalize_filename(result_file)
        
        if expected_normalized == result_normalized:
            return True, i + 1, result
    
    return False, None, None

def run_evaluation(test_set, apis):
    """Run evaluation on all APIs"""
    print("\n" + "=" * 80)
    print("🚀 STARTING EVALUATION")
    print("=" * 80)
    print(f"📊 Test set size: {len(test_set)}")
    print(f"🔍 APIs to test: {len(apis)}")
    print(f"⚙️  Top K: {TOP_K}")
    print(f"⚙️  Semantic weight: {SEMANTIC_WEIGHT}")
    print(f"⚙️  Keyword weight: {KEYWORD_WEIGHT}")
    print("=" * 80)
    
    # Results storage
    all_results = {
        "test_set_size": len(test_set),
        "top_k": TOP_K,
        "semantic_weight": SEMANTIC_WEIGHT,
        "keyword_weight": KEYWORD_WEIGHT,
        "apis": {}
    }
    
    for api in apis:
        print(f"\n{'='*80}")
        print(f"🧪 Testing API: {api['name']} (Port {api['port']})")
        print(f"{'='*80}")
        
        api_results = {
            "name": api["name"],
            "url": api["url"],
            "queries": [],
            "stats": {
                "total": len(test_set),
                "top1_success": 0,
                "top3_success": 0,
                "top5_success": 0,
                "errors": 0,
                "total_time_ms": 0
            }
        }
        
        start_time = time.time()
        
        for i, test_query in enumerate(test_set, 1):
            query_text = test_query["text"]
            expected_book = test_query["source_file"]
            
            # Query API
            response = query_api(api, query_text, TOP_K, SEMANTIC_WEIGHT, KEYWORD_WEIGHT)
            
            # Check success at different ranks
            if "error" not in response:
                top1_success, top1_rank, top1_result = check_success(expected_book, response, 1)
                top3_success, top3_rank, top3_result = check_success(expected_book, response, 3)
                top5_success, top5_rank, top5_result = check_success(expected_book, response, 5)
                
                if top1_success:
                    api_results["stats"]["top1_success"] += 1
                if top3_success:
                    api_results["stats"]["top3_success"] += 1
                if top5_success:
                    api_results["stats"]["top5_success"] += 1
                
                # Store query result
                api_results["queries"].append({
                    "query_id": i,
                    "expected_book": expected_book,
                    "top1_success": top1_success,
                    "top3_success": top3_success,
                    "top5_success": top5_success,
                    "found_at_rank": top5_rank if top5_success else None,
                    "top_result": response["results"][0]["source_file"] if response.get("results") else None,
                    "search_time_ms": response.get("search_ms", 0)
                })
                
                api_results["stats"]["total_time_ms"] += response.get("total_ms", 0)
            else:
                api_results["stats"]["errors"] += 1
                api_results["queries"].append({
                    "query_id": i,
                    "expected_book": expected_book,
                    "error": response["error"]
                })
            
            # Progress update
            if i % 50 == 0:
                elapsed = time.time() - start_time
                queries_per_sec = i / elapsed if elapsed > 0 else 0
                print(f"   Progress: {i}/{len(test_set)} queries | "
                      f"Top-1: {api_results['stats']['top1_success']}/{i} "
                      f"({100*api_results['stats']['top1_success']/i:.1f}%) | "
                      f"Speed: {queries_per_sec:.1f} q/s")
        
        # Calculate final stats
        total_valid = api_results["stats"]["total"] - api_results["stats"]["errors"]
        api_results["stats"]["top1_accuracy"] = (api_results["stats"]["top1_success"] / total_valid * 100) if total_valid > 0 else 0
        api_results["stats"]["top3_accuracy"] = (api_results["stats"]["top3_success"] / total_valid * 100) if total_valid > 0 else 0
        api_results["stats"]["top5_accuracy"] = (api_results["stats"]["top5_success"] / total_valid * 100) if total_valid > 0 else 0
        api_results["stats"]["avg_time_ms"] = api_results["stats"]["total_time_ms"] / total_valid if total_valid > 0 else 0
        
        all_results["apis"][api["name"]] = api_results
        
        # Print summary
        print(f"\n   ✅ Completed {api['name']}:")
        print(f"      Top-1 Accuracy: {api_results['stats']['top1_accuracy']:.2f}%")
        print(f"      Top-3 Accuracy: {api_results['stats']['top3_accuracy']:.2f}%")
        print(f"      Top-5 Accuracy: {api_results['stats']['top5_accuracy']:.2f}%")
        print(f"      Errors: {api_results['stats']['errors']}")
        print(f"      Avg Time: {api_results['stats']['avg_time_ms']:.1f}ms")
    
    return all_results

def generate_report(results):
    """Generate detailed text report"""
    report_lines = []
    
    report_lines.append("=" * 80)
    report_lines.append("BANGLA TEST SET EVALUATION REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append(f"Test Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Test Set Size: {results['test_set_size']} queries")
    report_lines.append(f"Top K: {results['top_k']}")
    report_lines.append(f"Semantic Weight: {results['semantic_weight']}")
    report_lines.append(f"Keyword Weight: {results['keyword_weight']}")
    report_lines.append("")
    
    report_lines.append("=" * 80)
    report_lines.append("SUMMARY - ALL APIS")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Create comparison table
    report_lines.append(f"{'API Name':<25} {'Top-1':<12} {'Top-3':<12} {'Top-5':<12} {'Avg Time':<12}")
    report_lines.append("-" * 80)
    
    for api_name, api_data in results["apis"].items():
        stats = api_data["stats"]
        report_lines.append(
            f"{api_name:<25} "
            f"{stats['top1_accuracy']:>6.2f}%     "
            f"{stats['top3_accuracy']:>6.2f}%     "
            f"{stats['top5_accuracy']:>6.2f}%     "
            f"{stats['avg_time_ms']:>6.1f}ms"
        )
    
    report_lines.append("")
    
    # Detailed results for each API
    for api_name, api_data in results["apis"].items():
        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append(f"DETAILED RESULTS: {api_name}")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        stats = api_data["stats"]
        report_lines.append(f"Total Queries: {stats['total']}")
        report_lines.append(f"Successful Queries: {stats['total'] - stats['errors']}")
        report_lines.append(f"Errors: {stats['errors']}")
        report_lines.append("")
        report_lines.append(f"Top-1 Success: {stats['top1_success']} / {stats['total']} ({stats['top1_accuracy']:.2f}%)")
        report_lines.append(f"Top-3 Success: {stats['top3_success']} / {stats['total']} ({stats['top3_accuracy']:.2f}%)")
        report_lines.append(f"Top-5 Success: {stats['top5_success']} / {stats['total']} ({stats['top5_accuracy']:.2f}%)")
        report_lines.append("")
        report_lines.append(f"Total Time: {stats['total_time_ms']:.1f}ms")
        report_lines.append(f"Average Time per Query: {stats['avg_time_ms']:.1f}ms")
        report_lines.append("")
        
        # Analyze failures (Top-1 failures)
        failures = [q for q in api_data["queries"] if not q.get("top1_success", False) and "error" not in q]
        if failures:
            report_lines.append(f"Top-1 Failures: {len(failures)}")
            report_lines.append("")
            report_lines.append("Sample failures (first 10):")
            for i, failure in enumerate(failures[:10], 1):
                report_lines.append(f"  {i}. Query #{failure['query_id']}")
                report_lines.append(f"     Expected: {failure['expected_book']}")
                report_lines.append(f"     Got: {failure['top_result']}")
                if failure.get('found_at_rank'):
                    report_lines.append(f"     Found at rank: {failure['found_at_rank']}")
                report_lines.append("")
    
    # Winner analysis
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("WINNER ANALYSIS")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    best_top1 = max(results["apis"].items(), key=lambda x: x[1]["stats"]["top1_accuracy"])
    best_top3 = max(results["apis"].items(), key=lambda x: x[1]["stats"]["top3_accuracy"])
    fastest = min(results["apis"].items(), key=lambda x: x[1]["stats"]["avg_time_ms"])
    
    report_lines.append(f"🏆 Best Top-1 Accuracy: {best_top1[0]} ({best_top1[1]['stats']['top1_accuracy']:.2f}%)")
    report_lines.append(f"🏆 Best Top-3 Accuracy: {best_top3[0]} ({best_top3[1]['stats']['top3_accuracy']:.2f}%)")
    report_lines.append(f"⚡ Fastest: {fastest[0]} ({fastest[1]['stats']['avg_time_ms']:.1f}ms)")
    report_lines.append("")
    
    return "\n".join(report_lines)

def main():
    print("🧪 BANGLA TEST SET EVALUATOR")
    print("=" * 80)
    
    # Load test set
    test_set = load_test_set(TEST_SET_FILE)
    
    # Run evaluation
    results = run_evaluation(test_set, APIS)
    
    # Save detailed results
    print(f"\n💾 Saving detailed results to: {RESULTS_FILE}")
    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Generate and save report
    print(f"📄 Generating report...")
    report = generate_report(results)
    
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"💾 Report saved to: {REPORT_FILE}")
    
    # Print summary to console
    print("\n" + "=" * 80)
    print("📊 FINAL RESULTS SUMMARY")
    print("=" * 80)
    print(f"\n{'API Name':<25} {'Top-1':<12} {'Top-3':<12} {'Top-5':<12} {'Avg Time':<12}")
    print("-" * 80)
    
    for api_name, api_data in results["apis"].items():
        stats = api_data["stats"]
        print(
            f"{api_name:<25} "
            f"{stats['top1_accuracy']:>6.2f}%     "
            f"{stats['top3_accuracy']:>6.2f}%     "
            f"{stats['top5_accuracy']:>6.2f}%     "
            f"{stats['avg_time_ms']:>6.1f}ms"
        )
    
    print("\n🎉 Evaluation complete!")
    print(f"📂 Detailed results: {RESULTS_FILE}")
    print(f"📄 Full report: {REPORT_FILE}")

if __name__ == "__main__":
    main()
