import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ScenariosSurveyComponent } from './scenarios-survey.component';

describe('ScenariosSurveyComponent', () => {
  let component: ScenariosSurveyComponent;
  let fixture: ComponentFixture<ScenariosSurveyComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ScenariosSurveyComponent]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(ScenariosSurveyComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
