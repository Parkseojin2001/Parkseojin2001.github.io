#!/usr/bin/env bash
#
# Create a new post, devlog post or project page from a template.
#
# Usage: bash tools/new.sh
# Env:   NEW_TODAY=YYYY-MM-DD  overrides today's date (used by tests)

set -eu

cd "$(dirname "$0")/.."

TODAY="${NEW_TODAY:-$(date +%F)}"
TEMPLATES="tools/templates"
NAME_RE='^[a-z0-9]+(-[a-z0-9]+)*$'
PATH_RE='^[a-z0-9-]+(/[a-z0-9-]+)*$'

if [ -t 1 ]; then
  GREEN=$'\033[32m' RED=$'\033[31m' BOLD=$'\033[1m' RESET=$'\033[0m'
else
  GREEN="" RED="" BOLD="" RESET=""
fi

ok() { printf '  %s✓%s %s\n' "$GREEN" "$RESET" "$1"; }
err() { printf '  %s✗%s %s\n' "$RED" "$RESET" "$1"; }
die() {
  err "$1"
  exit 1
}

# ask <prompt>: reads one line into REPLY; stops when input ends
ask() {
  printf '%s' "$1"
  if ! IFS= read -r REPLY; then
    echo
    die "입력이 끝났습니다"
  fi
}

# pick_line <list> <n>: prints the n-th line of a newline-separated list
pick_line() {
  printf '%s\n' "$1" | sed -n "${2}p"
}

choose_type() {
  printf '\n  %s어떤 글을 만들까요?%s\n' "$BOLD" "$RESET"
  echo "    1) 일반 포스트"
  echo "    2) 개발 기록 (프로젝트 진행 중 글)"
  echo "    3) 프로젝트 (완성본 페이지)"
  while true; do
    ask "  > "
    case "$REPLY" in
      1 | 2 | 3) TYPE="$REPLY" && return ;;
      *) err "1, 2, 3 중에서 골라 주세요" ;;
    esac
  done
}

choose_category() {
  local dirs count i
  dirs=$(find _posts -type f -name '*.md' -not -path '_posts/projects/*' 2>/dev/null |
    sed 's|/[^/]*$||; s|^_posts/\{0,1\}||' | grep -v '^_posts$' | grep -v '^$' | sort -u || true)
  count=$(printf '%s' "$dirs" | grep -c . || true)

  printf '\n  %s카테고리 폴더를 고르세요%s (번호, 또는 새 경로 예: ai-agent/langgraph)\n' "$BOLD" "$RESET"
  i=1
  while [ "$i" -le "$count" ]; do
    printf '    %d) %s\n' "$i" "$(pick_line "$dirs" "$i")"
    i=$((i + 1))
  done

  while true; do
    ask "  > "
    if [[ "$REPLY" =~ ^[0-9]+$ ]] && [ "$REPLY" -ge 1 ] && [ "$REPLY" -le "$count" ]; then
      FOLDER=$(pick_line "$dirs" "$REPLY")
      return
    elif [[ "$REPLY" =~ $PATH_RE ]] && [[ ! "$REPLY" =~ ^projects(/|$) ]]; then
      FOLDER="$REPLY"
      return
    fi
    err "목록의 번호나 영어 소문자 경로(예: ai-agent/langgraph)를 입력해 주세요"
  done
}

choose_project() {
  local names count i n title
  names=$(find _projects -maxdepth 1 -type f -name '*.md' 2>/dev/null | sed 's|^_projects/||; s|\.md$||' | sort || true)
  count=$(printf '%s' "$names" | grep -c . || true)
  [ "$count" -gt 0 ] || die "프로젝트가 없습니다. 먼저 3) 프로젝트를 만드세요"

  printf '\n  %s어떤 프로젝트의 기록인가요?%s\n' "$BOLD" "$RESET"
  i=1
  while [ "$i" -le "$count" ]; do
    n=$(pick_line "$names" "$i")
    title=$(sed -n 's/^title:[[:space:]]*//p' "_projects/$n.md" | head -n 1 | sed 's/[[:space:]]*#.*$//; s/^"//; s/"$//')
    printf '    %d) %-20s (%s)\n' "$i" "$n" "${title:-제목 없음}"
    i=$((i + 1))
  done

  while true; do
    ask "  > "
    if [[ "$REPLY" =~ ^[0-9]+$ ]] && [ "$REPLY" -ge 1 ] && [ "$REPLY" -le "$count" ]; then
      PROJECT=$(pick_line "$names" "$REPLY")
      return
    fi
    err "목록의 번호를 입력해 주세요"
  done
}

choose_name() {
  echo
  while true; do
    ask "  파일 이름 (영어 소문자, 숫자, 하이픈): "
    if [[ "$REPLY" =~ $NAME_RE ]]; then
      NAME="$REPLY"
      return
    fi
    err "영어 소문자, 숫자, 하이픈만 사용할 수 있습니다 (예: vector-db-comparison)"
  done
}

# render <template> <dest>
render() {
  sed -e "s|{{DATE}}|$TODAY|g" \
    -e "s|{{PERMALINK}}|${PERMALINK:-}|g" \
    -e "s|{{PROJECT}}|${PROJECT:-}|g" \
    -e "s|{{NAME}}|$NAME|g" \
    "$TEMPLATES/$1" >"$2"
}

main() {
  choose_type
  case "$TYPE" in
    1) choose_category ;;
    2) choose_project ;;
  esac
  choose_name

  case "$TYPE" in
    1)
      FILE="_posts/$FOLDER/$TODAY-$NAME.md"
      IMG_DIR="assets/img/post/$FOLDER"
      PERMALINK="/$FOLDER/$NAME/"
      TEMPLATE="post.md"
      ;;
    2)
      FILE="_posts/projects/$PROJECT/$TODAY-$NAME.md"
      IMG_DIR="assets/img/post/projects/$PROJECT"
      PERMALINK="/projects/$PROJECT/$NAME/"
      TEMPLATE="devlog.md"
      ;;
    3)
      FILE="_projects/$NAME.md"
      IMG_DIR="assets/img/projects/$NAME"
      PERMALINK="/projects/$NAME/"
      TEMPLATE="project.md"
      ;;
  esac

  [ ! -e "$FILE" ] || die "이미 있는 파일입니다: $FILE"

  # 같은 이름을 다른 날 만들면 파일은 달라도 주소가 겹쳐 한쪽이 덮어써진다
  if [ "$TYPE" != 3 ] && grep -rqE "^permalink:[[:space:]]*${PERMALINK}([[:space:]]|\$)" _posts 2>/dev/null; then
    die "같은 주소가 이미 있습니다: $PERMALINK"
  fi

  mkdir -p "$(dirname "$FILE")"
  render "$TEMPLATE" "$FILE"

  if [ -d "$IMG_DIR" ]; then
    IMG_STATE="(이미 있음)"
  else
    mkdir -p "$IMG_DIR"
    IMG_STATE="(새로 만듦)"
  fi

  echo
  ok "글 파일      $FILE"
  ok "이미지 폴더  $IMG_DIR/   $IMG_STATE"
  ok "주소         $PERMALINK"
  if [ "$TYPE" = 2 ]; then
    ok "연결         project: $PROJECT → 프로젝트 페이지 ./devlog에 자동 표시"
  fi

  echo
  printf '  %s다음 할 일%s\n' "$BOLD" "$RESET"
  case "$TYPE" in
    3) echo "    1. [필수] 항목 채우기: title, summary, period, role, stack" ;;
    *) echo "    1. [필수] 항목 채우기: title, categories (description도 권장)" ;;
  esac
  echo "    2. 이미지는 위 폴더에 넣고 이렇게 사용:"
  echo "       ![설명](/$IMG_DIR/파일명.png)"
  echo
}

main
