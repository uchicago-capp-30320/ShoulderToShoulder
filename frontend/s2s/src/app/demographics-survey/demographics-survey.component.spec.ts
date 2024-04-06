import { ComponentFixture, TestBed } from '@angular/core/testing';

import { DemographicsSurveyComponent } from './demographics-survey.component';

describe('DemographicsSurveyComponent', () => {
  let component: DemographicsSurveyComponent;
  let fixture: ComponentFixture<DemographicsSurveyComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [DemographicsSurveyComponent]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(DemographicsSurveyComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
