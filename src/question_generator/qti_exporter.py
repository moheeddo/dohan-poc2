"""
QTI (Question and Test Interoperability) 표준 내보내기

IMS QTI 2.1 호환 XML 형식으로 문항을 내보내어
CBT (Computer-Based Testing) 시스템과 연동.

지원 형식:
- QTI 2.1 XML (IMS Global 표준)
- JSON (내부 시스템용)
- CSV (간이 내보내기)

참고:
- IMS QTI 2.1: https://www.imsglobal.org/question/qtiv2p1/imsqti_implv2p1.html
- 한수원 CBT 시스템은 QTI 2.1 호환 LMS와 연동
"""
import json
import csv
import io
from datetime import datetime
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom.minidom import parseString


class QTIExporter:
    """QTI 2.1 표준 문항 내보내기"""

    @staticmethod
    def question_to_qti(question: dict, index: int = 1) -> Element:
        """단일 문항을 QTI 2.1 assessmentItem으로 변환"""
        qid = question.get("question_id", f"Q{index:03d}")
        q_text = question.get("question_text", "")
        options = question.get("options", {})
        correct = question.get("correct_answer", "A")
        explanation = question.get("explanation", "")
        bloom = question.get("bloom_level", "")
        scenario = question.get("scenario", "")
        safety = question.get("safety_significance", {}).get("grade", "general")

        # assessmentItem 루트
        item = Element("assessmentItem", {
            "xmlns": "http://www.imsglobal.org/xsd/imsqti_v2p1",
            "identifier": qid,
            "title": f"{qid} - {bloom}",
            "adaptive": "false",
            "timeDependent": "false",
        })

        # responseDeclaration
        resp_decl = SubElement(item, "responseDeclaration", {
            "identifier": "RESPONSE",
            "cardinality": "single",
            "baseType": "identifier",
        })
        correct_resp = SubElement(resp_decl, "correctResponse")
        value_el = SubElement(correct_resp, "value")
        value_el.text = f"choice_{correct}"

        # outcomeDeclaration
        outcome = SubElement(item, "outcomeDeclaration", {
            "identifier": "SCORE",
            "cardinality": "single",
            "baseType": "float",
        })
        default_val = SubElement(outcome, "defaultValue")
        val_el = SubElement(default_val, "value")
        val_el.text = "0"

        # itemBody
        body = SubElement(item, "itemBody")

        # 메타데이터 (커스텀)
        if bloom or safety:
            meta_div = SubElement(body, "div", {"class": "metadata"})
            if bloom:
                bloom_p = SubElement(meta_div, "p", {"class": "bloom-level"})
                bloom_p.text = f"[Bloom: {bloom}]"
            if safety != "general":
                safety_p = SubElement(meta_div, "p", {"class": "safety-grade"})
                safety_p.text = f"[Safety: {safety}]"

        # 시나리오 (있는 경우)
        if scenario:
            scenario_div = SubElement(body, "div", {"class": "scenario"})
            scenario_p = SubElement(scenario_div, "p")
            scenario_p.text = scenario

        # 문항 텍스트
        prompt_div = SubElement(body, "div", {"class": "prompt"})
        prompt_p = SubElement(prompt_div, "p")
        prompt_p.text = q_text

        # 선택지
        interaction = SubElement(body, "choiceInteraction", {
            "responseIdentifier": "RESPONSE",
            "shuffle": "false",
            "maxChoices": "1",
        })

        for key in sorted(options.keys()):
            choice = SubElement(interaction, "simpleChoice", {
                "identifier": f"choice_{key}",
            })
            choice.text = f"{key}. {options[key]}"

        # responseProcessing (정답 매칭)
        resp_proc = SubElement(item, "responseProcessing", {
            "template": "http://www.imsglobal.org/question/qti_v2p1/rptemplates/match_correct",
        })

        # modalFeedback (해설)
        if explanation:
            feedback = SubElement(item, "modalFeedback", {
                "outcomeIdentifier": "SCORE",
                "identifier": "EXPLANATION",
                "showHide": "show",
            })
            feedback.text = explanation

        return item

    @classmethod
    def export_to_qti_xml(cls, questions: list[dict], title: str = "KHNP Question Bank") -> str:
        """문항 세트를 QTI 2.1 XML 문자열로 내보내기"""
        # assessmentTest wrapper
        root = Element("assessmentTest", {
            "xmlns": "http://www.imsglobal.org/xsd/imsqti_v2p1",
            "identifier": f"test_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "title": title,
        })

        # testPart
        test_part = SubElement(root, "testPart", {
            "identifier": "part1",
            "navigationMode": "linear",
            "submissionMode": "individual",
        })

        # assessmentSection
        section = SubElement(test_part, "assessmentSection", {
            "identifier": "section1",
            "title": title,
            "visible": "true",
        })

        for i, q in enumerate(questions, 1):
            item = cls.question_to_qti(q, i)
            # assessmentItemRef로 참조
            ref = SubElement(section, "assessmentItemRef", {
                "identifier": q.get("question_id", f"Q{i:03d}"),
                "href": f"items/{q.get('question_id', f'Q{i:03d}')}.xml",
            })

        xml_str = tostring(root, encoding="unicode")
        pretty = parseString(xml_str).toprettyxml(indent="  ")
        # XML declaration 중복 제거
        lines = pretty.split("\n")
        if lines and lines[0].startswith("<?xml"):
            lines[0] = '<?xml version="1.0" encoding="UTF-8"?>'
        return "\n".join(lines)

    @classmethod
    def export_items_xml(cls, questions: list[dict]) -> dict[str, str]:
        """개별 문항 XML 딕셔너리 반환 {question_id: xml_string}"""
        items = {}
        for i, q in enumerate(questions, 1):
            item = cls.question_to_qti(q, i)
            xml_str = tostring(item, encoding="unicode")
            pretty = parseString(xml_str).toprettyxml(indent="  ")
            qid = q.get("question_id", f"Q{i:03d}")
            items[qid] = pretty
        return items

    @staticmethod
    def export_to_csv(questions: list[dict]) -> str:
        """간이 CSV 내보내기 (LMS 일괄 업로드용)"""
        output = io.StringIO()
        writer = csv.writer(output)

        # 헤더
        writer.writerow([
            "question_id", "bloom_level", "safety_grade",
            "scenario", "question_text",
            "option_A", "option_B", "option_C", "option_D",
            "correct_answer", "explanation",
            "learning_objective", "difficulty_estimate",
        ])

        for q in questions:
            options = q.get("options", {})
            safety = q.get("safety_significance", {}).get("grade", "general")
            difficulty = q.get("estimated_item_analysis", {}).get("difficulty_category", "")

            writer.writerow([
                q.get("question_id", ""),
                q.get("bloom_level", ""),
                safety,
                q.get("scenario", ""),
                q.get("question_text", ""),
                options.get("A", ""),
                options.get("B", ""),
                options.get("C", ""),
                options.get("D", ""),
                q.get("correct_answer", ""),
                q.get("explanation", ""),
                q.get("learning_objective", ""),
                difficulty,
            ])

        return output.getvalue()

    @staticmethod
    def export_to_json(questions: list[dict], include_metadata: bool = True) -> str:
        """구조화된 JSON 내보내기"""
        export_data = {
            "format": "khnp-qbank-v3",
            "exported_at": datetime.now().isoformat(),
            "total_questions": len(questions),
            "questions": [],
        }

        for q in questions:
            item = {
                "question_id": q.get("question_id", ""),
                "bloom_level": q.get("bloom_level", ""),
                "question_text": q.get("question_text", ""),
                "options": q.get("options", {}),
                "correct_answer": q.get("correct_answer", ""),
                "explanation": q.get("explanation", ""),
            }

            if q.get("scenario"):
                item["scenario"] = q["scenario"]

            if include_metadata:
                item["learning_objective"] = q.get("learning_objective", "")
                item["distractor_rationale"] = q.get("distractor_rationale", {})
                item["safety_significance"] = q.get("safety_significance", {})
                item["estimated_item_analysis"] = q.get("estimated_item_analysis", {})

            export_data["questions"].append(item)

        return json.dumps(export_data, ensure_ascii=False, indent=2)

    @classmethod
    def save_export(
        cls,
        questions: list[dict],
        output_dir: str,
        formats: list[str] = None,
        title: str = "KHNP Question Bank",
    ) -> dict[str, str]:
        """여러 형식으로 한번에 저장"""
        if formats is None:
            formats = ["qti", "json", "csv"]

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        saved = {}

        if "qti" in formats:
            qti_xml = cls.export_to_qti_xml(questions, title)
            fpath = output_path / f"qbank_qti_{timestamp}.xml"
            fpath.write_text(qti_xml, encoding="utf-8")
            saved["qti"] = str(fpath)

        if "json" in formats:
            json_str = cls.export_to_json(questions)
            fpath = output_path / f"qbank_{timestamp}.json"
            fpath.write_text(json_str, encoding="utf-8")
            saved["json"] = str(fpath)

        if "csv" in formats:
            csv_str = cls.export_to_csv(questions)
            fpath = output_path / f"qbank_{timestamp}.csv"
            fpath.write_text(csv_str, encoding="utf-8")
            saved["csv"] = str(fpath)

        return saved
