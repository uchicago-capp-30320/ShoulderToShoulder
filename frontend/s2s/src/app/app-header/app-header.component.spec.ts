import { ComponentFixture, TestBed } from '@angular/core/testing';

import { AppHeaderComponent } from './app-header.component';
import { NavbarComponent } from '../navbar/navbar.component';
import { MenubarModule } from 'primeng/menubar';
import { ActivatedRoute } from '@angular/router';

describe('AppHeaderComponent', () => {
  let component: AppHeaderComponent;
  let fixture: ComponentFixture<AppHeaderComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [AppHeaderComponent, NavbarComponent],
      imports: [MenubarModule],
      providers: [
        { provide: ActivatedRoute, useValue: {} } // Mock ActivatedRoute without any specific data
      ]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(AppHeaderComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
