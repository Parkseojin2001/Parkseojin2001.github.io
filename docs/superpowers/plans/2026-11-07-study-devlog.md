# Study and Devlog Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan inline.

**Goal:** Replace Blog navigation with separate Study and Devlog lists.

**Architecture:** Classify posts using explicit `section` first, then the existing `project` relationship. Generate independently paginated lists using the existing list design. Preserve post URLs and legacy Blog pagination.

**Tech Stack:** Jekyll, Ruby, Liquid, SCSS.

## Constraints

- Navigation order: Projects, Study, Devlog, About.
- Default posts to Study; project-linked posts default to Devlog.
- Support `section: study` and `section: devlog` overrides.
- Exclude hidden/unpublished posts according to Jekyll publication settings.
- Keep category navigation, pin ordering, post links and mobile navigation working.

## Steps

- [x] Add post classification and section-specific pagination.
- [x] Reuse the existing list layout, scoped category tree and paginator.
- [x] Update navigation, home links, 404 shortcut and authoring templates.
- [x] Build and check section membership, pagination, active menus, empty lists and legacy links.

## Validation

Production build succeeds; Devlog displays its empty-state message. The unpublished sample build verifies project-linked posts appear in Devlog, Study pagination contains all remaining posts exactly once, navigation selects the right section, and legacy Blog links still work. Section and Blog checks: 13 runs, 194 assertions, no failures.

## Writing posts

`tools/new.sh` creates Study posts with `section: study` and development records with `section: devlog`. Existing posts without this field default to Study unless they have a non-empty `project` link. An explicit section overrides that default, so learning notes can link to a project while remaining in Study. Existing post URLs do not change.
