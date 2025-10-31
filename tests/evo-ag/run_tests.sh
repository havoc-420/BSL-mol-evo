#!/bin/bash

# MoleculeEvolver 测试运行脚本
# 使用方法: ./run_tests.sh [test_type]
# test_type: all | basic | extended | consistency

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# 打印带颜色的信息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_result() {
    if [ "$1" -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $2"
    else
        echo -e "${RED}✗${NC} $2"
    fi
}

print_header() {
    echo -e "${BOLD}==============================================${NC}"
    echo -e "${BOLD}$1${NC}"
    echo -e "${BOLD}==============================================${NC}"
}

# 显示帮助信息
show_help() {
    echo "MoleculeEvolver 测试运行脚本"
    echo "使用方法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  all          运行所有测试 (默认)"
    echo "  basic        运行基础功能测试"
    echo "  extended     运行扩展测试"
    echo "  consistency  运行一致性测试"
    echo "  help         显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 all       # 运行所有测试"
    echo "  $0 basic     # 运行基础测试"
}

# 运行测试的函数
run_test() {
    local test_file=$1
    local test_name=$2
    
    print_info "运行 ${test_name}..."
    cd "$(dirname "$0")" || exit 1
    
    # 设置测试数据根目录环境变量
    export TEST_DATA_ROOT="$(pwd)/testcases"
    
    # 运行测试
    python "${test_file}"
    local result=$?
    
    # 检查返回码
    if [ $result -eq 0 ]; then
        print_success "${test_name} 全部通过"
    else
        print_error "${test_name} 存在失败用例"
    fi
    
    return $result
}

# 运行所有测试
run_all_tests() {
    print_info "运行所有测试..."
    local all_result=0
    
    # 运行基础测试
    echo
    print_header "运行基础测试"
    run_test "tests/test_basic.py" "基础测试"
    local basic_result=$?
    print_result $basic_result "基础测试"
    
    # 运行扩展测试
    echo
    print_header "运行扩展测试"
    run_test "tests/test_extended.py" "扩展测试"
    local extended_result=$?
    print_result $extended_result "扩展测试"
    
    # 运行一致性测试
    echo
    print_header "运行一致性测试"
    run_test "tests/test_consistency.py" "一致性测试"
    local consistency_result=$?
    print_result $consistency_result "一致性测试"
    
    echo
    print_header "测试汇总报告"
    if [ $basic_result -eq 0 ] && [ $extended_result -eq 0 ] && [ $consistency_result -eq 0 ]; then
        print_success "所有测试已通过!"
        return 0
    else
        if [ $basic_result -ne 0 ]; then
            print_error "基础测试存在失败项"
        fi
        if [ $extended_result -ne 0 ]; then
            print_error "扩展测试存在失败项"
        fi
        if [ $consistency_result -ne 0 ]; then
            print_error "一致性测试存在失败项"
        fi
        print_error "部分测试失败，请检查以上输出"
        return 1
    fi
}

# 主逻辑
main() {
    local test_type=${1:-all}
    
    case "${test_type}" in
        all)
            run_all_tests
            ;;
        basic)
            echo
            print_header "运行基础测试"
            run_test "tests/test_basic.py" "基础测试"
            ;;
        extended)
            echo
            print_header "运行扩展测试"
            run_test "tests/test_extended.py" "扩展测试"
            ;;
        consistency)
            echo
            print_header "运行一致性测试"
            run_test "tests/test_consistency.py" "一致性测试"
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            print_error "未知选项: $test_type"
            show_help
            exit 1
            ;;
    esac
}

# 执行主函数
main "$@"