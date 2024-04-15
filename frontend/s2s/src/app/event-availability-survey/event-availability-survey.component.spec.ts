import { ComponentFixture, TestBed } from '@angular/core/testing';
import { FormsModule, FormGroupDirective, ReactiveFormsModule } from '@angular/forms';

// primeng
import { CheckboxModule } from 'primeng/checkbox';
import { MultiSelectModule } from 'primeng/multiselect';

// components and services
import { EventAvailabilitySurveyComponent } from './event-availability-survey.component';

describe('EventAvailabilitySurveyComponent', () => {
  let component: EventAvailabilitySurveyComponent;
  let fixture: ComponentFixture<EventAvailabilitySurveyComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [
        EventAvailabilitySurveyComponent
      ],
      imports: [
        FormsModule,
        ReactiveFormsModule,
        CheckboxModule,
        MultiSelectModule
      ],
      providers: [
        FormGroupDirective
      ],
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(EventAvailabilitySurveyComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should correct initialize the general availability form', () => {
    const form = component.generalAvailabilityForm;
    expect(form).toBeTruthy()
    expect(form.get('weekdayEarlyMorning')).toBeTruthy();
    expect(form.get('weekdayMorning')).toBeTruthy();
    expect(form.get('weekdayAfternoon')).toBeTruthy();
    expect(form.get('weekdayEvening')).toBeTruthy();
    expect(form.get('weekdayNight')).toBeTruthy();
    expect(form.get('weekdayLateNight')).toBeTruthy();
    expect(form.get('weekdayUnavailable')).toBeTruthy();
    expect(form.get('weekendEarlyMorning')).toBeTruthy();
    expect(form.get('weekendMorning')).toBeTruthy();
    expect(form.get('weekendAfternoon')).toBeTruthy();
    expect(form.get('weekendEvening')).toBeTruthy();
    expect(form.get('weekendNight')).toBeTruthy();
    expect(form.get('weekendLateNight')).toBeTruthy();
    expect(form.get('weekendUnavailable')).toBeTruthy();
  });

  it('should correctly disable all general availability controls for weekdays when unavailable is chosen', () => {
    const form = component.generalAvailabilityForm;
    const weekdayUnavailable = form.get('weekdayUnavailable');
    weekdayUnavailable?.setValue(true);

    component.weekdayGeneralAvailabilityControls.forEach(control => {
      if (!control.toLowerCase().includes('unavailable')) {
        expect(component.getWeekdayDisabledState(control)).toBeTrue();
      }
    });
      
  });

  it('should correctly disable unavailable control for weekdays when any other control is chosen', () => {
    const form = component.generalAvailabilityForm;
    component.weekdayGeneralAvailabilityControls.forEach(control => {
      if (!control.toLowerCase().includes('unavailable')) {
        form.get(control)?.setValue(true);
        expect(component.getWeekdayDisabledState('weekdayUnavailable')).toBeTrue();
        form.get(control)?.setValue(false);
      }
    });
  });

  it('should correctly disable all general availability controls for weekends when unavailable is chosen', () => {
    const form = component.generalAvailabilityForm;
    const weekendUnavailable = form.get('weekendUnavailable');
    weekendUnavailable?.setValue(true);

    fixture.detectChanges();
    component.weekendGeneralAvailabilityControls.forEach(control => {
      if (!control.toLowerCase().includes('unavailable')) {
        expect(component.getWeekendDisabledState(control)).toBeTrue();
      }
    });
  });

  it('should correctly disable unavailable control for weekends when any other control is chosen', () => {
    const form = component.generalAvailabilityForm;
    component.weekendGeneralAvailabilityControls.forEach(control => {
      if (!control.toLowerCase().includes('unavailable')) {
        form.get(control)?.setValue(true);
        expect(component.getWeekendDisabledState('weekendUnavailable')).toBeTrue();
        form.get(control)?.setValue(false);
      }
    });
  });

  it('should correctly add a time range to the selected day', () => {
    const form = component.userService.eventAvailabilityForm;
    const timeRange = 'Early morning (5-8a)';
    const correctTimes = [5, 6, 7, 8];
    const days = ['mondayTimes'];

    component.addTimeRange(timeRange, days);
    const mondayTimes = form.get('mondayTimes')?.value;
    expect(mondayTimes).toEqual(correctTimes);
  });

  it('should correctly remove a time from the selected day', () => {
    const form = component.userService.eventAvailabilityForm;
    const timeRange = 'Early morning (5-8a)';
    const days = ['mondayTimes'];

    form.get('mondayTimes')?.setValue([1, 2, 3, 4, 5, 6, 7, 8]);
    component.removeTimeRange(timeRange, days);
    const mondayTimes = form.get('mondayTimes')?.value;
    expect(mondayTimes).toEqual([1, 2, 3, 4]);
  });
});
