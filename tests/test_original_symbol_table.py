"""Synthetic provenance/coverage controls, never benchmark labels."""
import copy
import json
from unittest import TestCase
from unittest.mock import patch
import numpy as np
from PIL import Image
from evidence_lab.source_symbol_table import parse_labels,candidate_from_labels,exact_symbol_plan,compose_original_symbol_candidate
from evidence_lab.source_symbol_table import label_messages

def fixture():
    plan={"grid":{"row_count":3,"column_count":4},"legend":[{"box_pixels":[i,20,i+1,21]} for i in range(3)],
        "cells":[{"row_index":r,"column_index":c,"anonymous_legend_index":(r+c)%3,
                  "glyph_box_pixels":[c,r,c+1,r+1]} for r in (1,2) for c in (1,2,3)]}
    labels={"columns":["Alpha model","Beta model","Gamma model"],"rows":["Efficiency","Durability"],
            "legend":["High","Medium","Low"]}
    return plan,labels

class OriginalSymbolTests(TestCase):
    def test_processor_contract_has_typed_content_in_every_role(self):
        plan,_=fixture()
        panels=[{"name":name,"path":name+".png"} for name in ("columns","rows","legend")]
        messages=label_messages(plan,panels)
        for message in messages:
            self.assertIsInstance(message["content"],list)
            self.assertTrue(all(item["type"] in ("text","image") for item in message["content"]))
        self.assertEqual([item["image"] for item in messages[1]["content"] if item["type"]=="image"],
                         [p["path"] for p in panels])

    def test_requested_named_axes_are_bound_without_extraneous_row(self):
        plan,labels=fixture();before=copy.deepcopy(plan)
        c=candidate_from_labels("Compare Efficiency for Alpha and Gamma models.",{"image_id":"synthetic"},plan,labels,{})
        self.assertEqual(c["answer"],"Efficiency: Alpha model = Low, Gamma model = Medium.")
        self.assertEqual([(x["row_index"],x["column_index"]) for x in c["requested_cell_facts"]],[(1,1),(1,3)])
        self.assertFalse(c["semantic_support_established"])
        self.assertEqual(plan,before)

    def test_full_name_and_non_strict_operator_are_not_rewritten(self):
        plan,labels=fixture();labels["legend"][0]="≥ 2 units"
        c=candidate_from_labels("Compare Durability for Alpha and Beta models.",{"image_id":"synthetic"},plan,labels,{})
        self.assertIn("≥ 2 units",c["answer"]);self.assertNotIn("> 2",c["answer"])

    def test_unmatched_axis_and_single_named_column_decline(self):
        plan,labels=fixture()
        for q in ("Compare Efficiency of Alpha.", "Compare Reliability of Alpha and Beta.", "Compare Efficiency of Delta and Beta."):
            self.assertIsNone(candidate_from_labels(q,{"image_id":"synthetic"},plan,labels,{}))

    def test_missing_or_duplicate_label_is_not_invented(self):
        plan,labels=fixture()
        for key in labels:
            for badlist in (labels[key][:-1],[""]+labels[key][1:],[labels[key][0]]*len(labels[key])):
                bad=copy.deepcopy(labels);bad[key]=badlist
                with self.assertRaises(ValueError):parse_labels(json.dumps(bad),plan)

    def test_invalid_json_keys_and_control_characters_reject(self):
        plan,labels=fixture()
        bad=copy.deepcopy(labels);bad["rows"][0]="Efficiency\nInvented"
        for raw in (json.dumps(bad),'{"rows":[],"rows":[],"columns":[],"legend":[]}',json.dumps({**labels,"answer":"unsupported"})):
            with self.assertRaises(ValueError):parse_labels(raw,plan)

    def test_no_unverified_live_context_triggers_model(self):
        class Unexpected:
            def __getattr__(self,k):raise AssertionError("No actual work expected")
        raw={"raw_output_for_validation":"retained"}
        out,report=compose_original_symbol_candidate("Question",raw,object(),visual=Unexpected(),reader=Unexpected(),
            processor=Unexpected(),budget=Unexpected(),config={},output="unused",prior_recovery={})
        self.assertIs(out,raw);self.assertFalse(report["eligible"]);self.assertEqual(report["actual_transcription_calls"],0)

    def _plan_control(self,alter=False,missing=False):
        pixels=np.full((30,30,3),255,dtype=np.uint8)
        for box in ([3,5,6,8],[13,5,16,8],[3,21,6,24],[13,21,16,24]):
            x0,y0,x1,y1=box;pixels[y0:y1,x0:x1]=0
        if alter:pixels[6,4]=255
        grid={"row_count":2,"column_count":3,"cells":[],
              "horizontal_rule_pixel_bands":[[0,1],[3,4],[10,11]],
              "vertical_rule_pixel_bands":[[0,1],[10,11],[20,21],[29,30]]}
        cells=[{"row_index":1,"column_index":i+1,"glyph_box_pixels":b,
                "anonymous_legend_index":i,"glyph_mask_sha256":"fixed"}
               for i,b in enumerate(([3,5,6,8],[13,5,16,8]))]
        if missing:cells.pop()
        proposal={"cell_matches":cells,"legend_prototypes":[
            {"box_pixels":[3,21,6,24],"mask_sha256":"fixed"},
            {"box_pixels":[13,21,16,24],"mask_sha256":"fixed"}]}
        with patch("evidence_lab.source_symbol_table.original_pixels",return_value=(Image.fromarray(pixels),{"native_size_pixels":[30,30]})),\
             patch("evidence_lab.source_symbol_table.grid_and_cells",return_value=(grid,None)),\
             patch("evidence_lab.source_symbol_table.match_legend",return_value=(proposal,{})):
            return exact_symbol_plan({},"unused")

    def test_near_match_cannot_bypass_exact_original_pixel_check(self):
        self.assertIsNotNone(self._plan_control())
        self.assertIsNone(self._plan_control(alter=True))

    def test_missing_grid_cell_cannot_be_omitted_from_coverage(self):
        self.assertIsNone(self._plan_control(missing=True))
