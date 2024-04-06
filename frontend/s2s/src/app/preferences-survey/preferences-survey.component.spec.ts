import { ComponentFixture, TestBed } from '@angular/core/testing';

import { PreferencesSurveyComponent } from './preferences-survey.component';

describe('PreferencesSurveyComponent', () => {
  let component: PreferencesSurveyComponent;
  let fixture: ComponentFixture<PreferencesSurveyComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [PreferencesSurveyComponent]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(PreferencesSurveyComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
