import { ComponentFixture, TestBed } from '@angular/core/testing';
import { HttpClientModule } from '@angular/common/http';
import { HttpClientTestingModule, HttpTestingController } from '@angular/common/http/testing';
import { FormsModule, ReactiveFormsModule, FormGroupDirective } from '@angular/forms';

// primeng
import { DropdownModule } from 'primeng/dropdown';
import { MultiSelectModule } from 'primeng/multiselect';
import { TooltipModule } from 'primeng/tooltip';

// components and services
import { PreferencesSurveyComponent } from './preferences-survey.component';

describe('PreferencesSurveyComponent', () => {
  let component: PreferencesSurveyComponent;
  let fixture: ComponentFixture<PreferencesSurveyComponent>;
  let httpController: HttpTestingController;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [
        PreferencesSurveyComponent
      ],
      imports: [
        HttpClientModule,
        DropdownModule,
        MultiSelectModule,
        FormsModule,
        ReactiveFormsModule,
        TooltipModule,
        HttpClientTestingModule
      ],
      providers: [
        FormGroupDirective
      ]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(PreferencesSurveyComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
    httpController = TestBed.inject(HttpTestingController);
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should create a string list of hobbies', () => {
    expect(component.hobbies).toBeTruthy();
    expect(component.hobbies.length).toBeGreaterThan(0);
    expect(component.leastInterestedHobbies).toBeTruthy();
    expect(component.leastInterestedHobbies.length).toBeGreaterThan(0);
    expect(component.mostInterestedHobbies).toBeTruthy();
    expect(component.mostInterestedHobbies.length).toBeGreaterThan(0);
  });
});
