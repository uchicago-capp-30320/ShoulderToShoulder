import { ComponentFixture, TestBed } from '@angular/core/testing';

import { EventAvailabilitySurveyComponent } from './event-availability-survey.component';

describe('EventAvailabilitySurveyComponent', () => {
  let component: EventAvailabilitySurveyComponent;
  let fixture: ComponentFixture<EventAvailabilitySurveyComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [EventAvailabilitySurveyComponent]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(EventAvailabilitySurveyComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
