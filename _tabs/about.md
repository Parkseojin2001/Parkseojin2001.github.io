---
layout: about
title: About
icon: fas fa-info-circle
order: 4
---

<section class="about-profile" aria-labelledby="about-name">
  <img class="about-avatar" src="{{ site.avatar | relative_url }}" alt="박서진 프로필" width="128" height="128">
  <div>
    <p class="about-eyebrow mono">DATA · BACKEND · AI</p>
    <h2 id="about-name">박서진 <span>Seojin Park</span></h2>
    <p class="about-intro">데이터 파이프라인과 백엔드를 만들고, 배운 내용과 문제를 해결한 과정을 기록합니다.</p>
    <p class="about-description">공부하며 이해한 내용과 개발하면서 겪은 문제를 기록합니다. 네이버 클라우드에서 영상 분석 기능의 성능을 측정하고 개선할 지점을 찾는 평가 파이프라인과 시각화 도구를 개발했습니다.</p>
  </div>
</section>

<section class="about-section" aria-labelledby="about-toolkit">
  <div class="about-section-heading">
    <span class="about-eyebrow mono" aria-hidden="true">01 / TOOLKIT</span>
    <h2 id="about-toolkit">사용하는 기술</h2>
  </div>
  <dl class="about-toolkit">
    <div><dt class="mono">LANGUAGE</dt><dd>Python</dd></div>
    <div><dt class="mono">AI / ML</dt><dd>PyTorch <span>·</span> Scikit-learn</dd></div>
    <div><dt class="mono">DATA</dt><dd>PostgreSQL <span>·</span> MongoDB <span>·</span> Milvus</dd></div>
    <div><dt class="mono">WORKFLOW</dt><dd>Airflow <span>·</span> Prefect</dd></div>
    <div><dt class="mono">BACKEND</dt><dd>FastAPI</dd></div>
    <div><dt class="mono">INFRA</dt><dd>Docker</dd></div>
  </dl>
</section>

<section class="about-section" aria-labelledby="about-experience">
  <div class="about-section-heading">
    <span class="about-eyebrow mono" aria-hidden="true">02 / EXPERIENCE</span>
    <h2 id="about-experience">경력 · 교육</h2>
  </div>
  <div class="about-timeline">
    <section class="about-event" aria-labelledby="about-naver-cloud">
      <div class="about-period mono"><time datetime="2026-04">2026.04</time> — <time datetime="2026-10">2026.10</time><span>6개월</span></div>
      <div class="about-event-content">
        <p class="about-role mono">INTERNSHIP</p>
        <h3 id="about-naver-cloud">네이버 클라우드</h3>
        <p class="about-team">Vision · Video Understanding</p>
        <p class="about-event-summary">영상 설명·장면 검색·하이라이트 추출 기능의 성능 개선을 위한 평가 파이프라인 개발</p>
        <ul class="about-work">
          <li>
            <h4>추론 연계 평가 파이프라인 · AutoCaption Eval</h4>
            <ul class="about-work-details">
              <li>추론 완료 감지 → 실험·버전·트랙별 LLM 평가를 연결하는 Prefect 기반 파이프라인 구축</li>
              <li>평가 상태·지표의 PostgreSQL 저장 및 대시보드 연동</li>
            </ul>
          </li>
          <li>
            <h4>LLM 기반 선정 기준 평가 · 프롬프트 엔지니어링</h4>
            <ul class="about-work-details">
              <li>하이라이트 선정 기준의 적합성을 판정하는 LLM 평가 로직 구현 및 GT 기반 프롬프트 개선</li>
              <li><strong>사람 판정과 평균 87.8% 일치</strong> (75건 · 6회 평가)</li>
            </ul>
          </li>
          <li>
            <h4>검색·하이라이트 벤치마크 파이프라인</h4>
            <ul class="about-work-details">
              <li>동일 GT 기반 검색 방식별 Precision@K·Recall@K·mAP@K·nDCG@K 산출 자동화</li>
              <li>기준 생성 → 장면·클립 검색 → 포착률 평가를 단일 명령으로 실행하는 통합 파이프라인 구축</li>
              <li>GT·예측 구간의 겹침과 누락을 확인하는 타임라인 시각화 도구 구현</li>
            </ul>
          </li>
        </ul>
      </div>
    </section>
    <section class="about-event" aria-labelledby="about-boostcamp">
      <div class="about-period mono"><time datetime="2025-09">2025.09</time> — <time datetime="2026-02">2026.02</time></div>
      <div class="about-event-content">
        <p class="about-role mono">EDUCATION</p>
        <h3 id="about-boostcamp">네이버 부스트캠프 AI Tech</h3>
        <p class="about-event-summary is-short">PyTorch 기초부터 CV·NLP·LLM, AI 서비스 개발까지 학습했습니다. 팀 프로젝트에서는 배운 내용을 바탕으로 함께 서비스를 구현하며 협업과 개발 과정을 경험했습니다.</p>
        <a class="about-text-link mono" href="{{ '/categories/naver-boostcamp-ai-tech/' | relative_url }}">부스트캠프 기록 보기 ↗</a>
      </div>
    </section>
    <section class="about-event" aria-labelledby="about-university">
      <div class="about-period mono"><time datetime="2021-03">2021.03</time> — <time datetime="2027-02">2027.02</time><span>졸업 예정</span></div>
      <div class="about-event-content">
        <p class="about-role mono">EDUCATION</p>
        <h3 id="about-university">세종대학교</h3>
        <p class="about-team">컴퓨터공학과</p>
      </div>
    </section>
  </div>
</section>

<section class="about-section about-contact" aria-labelledby="about-contact-title">
  <div class="about-section-heading">
    <span class="about-eyebrow mono" aria-hidden="true">03 / CONTACT</span>
    <h2 id="about-contact-title">연락처</h2>
  </div>
  <div class="about-contact-links">
    <a href="mailto:{{ site.social.email }}"><span class="mono">EMAIL ↗</span>{{ site.social.email }}</a>
    <a href="https://github.com/{{ site.github.username }}"><span class="mono">GITHUB ↗</span>{{ site.github.username }}</a>
  </div>
</section>
